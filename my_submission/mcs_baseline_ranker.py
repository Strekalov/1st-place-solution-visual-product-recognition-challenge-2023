import yaml
import torch
from tqdm import tqdm

import sys

sys.path.append("./MCS2023_baseline/")

from data_utils.augmentations import get_val_query_aug, get_val_gallery_aug
from data_utils.dataset import SubmissionDataset
from utils import convert_dict_to_tuple
from models.wb_net import Embedder, WBNet, Trunk


def average_query_expansion(query_vecs, reference_vecs, top_k=3):
    """
    Average Query Expansion (AQE)
    Ondrej Chum, et al. "Total Recall: Automatic Query Expansion with a Generative Feature Model for Object Retrieval,"
    International Conference of Computer Vision. 2007.
    https://www.robots.ox.ac.uk/~vgg/publications/papers/chum07b.pdf
    """
    # Query augmentation
    sim_mat = torch.cdist(query_vecs, reference_vecs)
    indices = torch.argsort(sim_mat, dim=1)

    top_k_ref_mean = torch.mean(reference_vecs[indices[:, :top_k], :], dim=1)
    query_vecs = torch.cat([query_vecs, top_k_ref_mean], dim=1)

    # Reference augmentation
    sim_mat = torch.cdist(reference_vecs, reference_vecs)
    indices = torch.argsort(sim_mat, dim=1)

    top_k_ref_mean = torch.mean(reference_vecs[indices[:, 1 : top_k + 1], :], dim=1)
    reference_vecs = torch.cat([reference_vecs, top_k_ref_mean], dim=1)

    return query_vecs, reference_vecs


def db_augmentation(query_vecs, reference_vecs, top_k=3):
    """
    Database-side feature augmentation (DBA)
    Albert Gordo, et al. "End-to-end Learning of Deep Visual Representations for Image Retrieval,"
    International Journal of Computer Vision. 2017.
    https://link.springer.com/article/10.1007/s11263-017-1016-8
    """
    weights = torch.logspace(0, -2.0, top_k + 1).cuda()

    # Query augmentation
    sim_mat = torch.cdist(query_vecs, reference_vecs)

    indices = torch.argsort(sim_mat, dim=1)

    top_k_ref = reference_vecs[indices[:, :top_k], :]
    query_vecs = torch.tensordot(
        weights,
        torch.cat([torch.unsqueeze(query_vecs, 1), top_k_ref], dim=1),
        dims=([0], [1]),
    )

    # Reference augmentation
    sim_mat = torch.cdist(reference_vecs, reference_vecs)
    indices = torch.argsort(sim_mat, dim=1)

    top_k_ref = reference_vecs[indices[:, : top_k + 1], :]
    reference_vecs = torch.tensordot(weights, top_k_ref, dims=([0], [1]))
    # reference_vecs = torch.tensordot(weights, torch.cat([torch.unsqueeze(query_vecs, 1), top_k_ref], dim=1), dims=([0], [1]))

    return query_vecs, reference_vecs


class MCS_BaseLine_Ranker:
    def __init__(self, dataset_path, gallery_csv_path, queries_csv_path):
        """
        Initialize your model here
        Inputs:
            dataset_path
            gallery_csv_path
            queries_csv_path
        """

        self.dataset_path = dataset_path
        self.gallery_csv_path = gallery_csv_path
        self.queries_csv_path = queries_csv_path
        self.max_predictions = 1000

        model_path = "./saved_full_models/wb_full.pt"
        self.batch_size = 64

        self.exp_cfg = "./MCS2023_baseline/config/wb.yml"
        self.inference_cfg = "./MCS2023_baseline/config/inference_config.yml"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(self.exp_cfg) as f:
            data = yaml.safe_load(f)
        self.exp_cfg = convert_dict_to_tuple(data)
        self.embedding_shape = 2048

        with open(self.inference_cfg) as f:
            data = yaml.safe_load(f)
        self.inference_cfg = convert_dict_to_tuple(data)

        with torch.jit.optimized_execution(True):
            self.model = torch.jit.load("best_model.pth")

            self.model.half()
            self.model.eval()
            self.model.to(self.device, memory_format=torch.channels_last)

        print("Weights are loaded!")

    def raise_aicrowd_error(self, msg):
        """Will be used by the evaluator to provide logs, DO NOT CHANGE"""
        raise NameError(msg)

    def predict_product_ranks(self):
        """
        This function should return a numpy array of shape `(num_queries, 1000)`.
        For ach query image your model will need to predict
        a set of 1000 unique gallery indexes, in order of best match first.

        Outputs:
            class_ranks - A 2D numpy array where the axes correspond to:
                          axis 0 - Batch size
                          axis 1 - An ordered rank list of matched image indexes, most confident prediction first
                            - maximum length of this should be 1000
                            - predictions above this limit will be dropped
                            - duplicates will be dropped such that the lowest index entry is preserved
        """

        gallery_dataset = SubmissionDataset(
            root=self.dataset_path,
            annotation_file=self.gallery_csv_path,
            transforms=get_val_gallery_aug(self.exp_cfg),
        )

        gallery_loader = torch.utils.data.DataLoader(
            gallery_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.inference_cfg.num_workers,
        )

        query_dataset = SubmissionDataset(
            root=self.dataset_path,
            annotation_file=self.queries_csv_path,
            transforms=get_val_query_aug(self.exp_cfg),
            with_bbox=True,
        )

        query_loader = torch.utils.data.DataLoader(
            query_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.inference_cfg.num_workers,
        )

        print("Calculating embeddings")
        gallery_embeddings = torch.zeros(
            (len(gallery_dataset), 2048),
            device="cuda",
            requires_grad=False,
            dtype=torch.half,
        )
        query_embeddings = torch.zeros(
            (len(query_dataset), 2048),
            device="cuda",
            requires_grad=False,
            dtype=torch.half,
        )

        with torch.no_grad():
            for i, images in tqdm(enumerate(gallery_loader), total=len(gallery_loader)):

                images = images.half().to(
                    self.device, memory_format=torch.channels_last
                )
                outputs = self.model(images)

                gallery_embeddings[
                    i * self.batch_size : (i * self.batch_size + self.batch_size), :
                ] = outputs

            for i, images in tqdm(enumerate(query_loader), total=len(query_loader)):
                images = images.half().to(
                    self.device, memory_format=torch.channels_last
                )

                outputs = self.model(images)

                query_embeddings[
                    i * self.batch_size : (i * self.batch_size + self.batch_size), :
                ] = outputs

        query_embeddings, gallery_embeddings = (
            query_embeddings.float(),
            gallery_embeddings.float(),
        )
        concat = torch.cat((query_embeddings, gallery_embeddings), dim=0)
        center = torch.mean(concat, dim=0)
        query_embeddings = query_embeddings - center
        gallery_embeddings = gallery_embeddings - center
        gallery_embeddings = torch.nn.functional.normalize(
            gallery_embeddings, p=2.0, dim=1
        )
        query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2.0, dim=1)

        query_embeddings, gallery_embeddings = db_augmentation(
            query_embeddings, gallery_embeddings, top_k=5
        )

        # query_distances = torch.cdist(query_embeddings, query_embeddings)
        # mask = query_distances < 0.4
        # for i in range(query_embeddings.shape[0]):
        #     query_embeddings[i] = query_embeddings[mask[i]].mean(dim=0)

        distances = torch.cdist(query_embeddings, gallery_embeddings)
        sorted_distances, sorted_indices = torch.sort(distances, dim=1)

        class_ranks = sorted_indices
        # take indexes of the most similar embeddings from the gallery_embeddings
        first_gallery_idx = class_ranks[:, 0]
        
        # take distance value of the most similar embeddings from the gallery_embeddings
        first_gallery_dstx = sorted_distances[:, 0]

        # take the most similar embedding from the gallery_embeddings by index
        rerank_embeddings1 = gallery_embeddings.index_select(0, first_gallery_idx)

        # if distance between most similar gallery and query < 0.8 
        # then add it to the new embeddings list for ranking (filter_rerank_embeddings1) 
        # else add embedding from query_embeddings
        mask1 = first_gallery_dstx < 0.8
        filter_rerank_embeddings1 = torch.where(
            mask1.view(-1, 1), rerank_embeddings1, query_embeddings
        )
        
        # averaging and ranking
        filter_rerank_embeddings = (
            0.5 * filter_rerank_embeddings1 + 0.5 * query_embeddings
        )
        distances = torch.cdist(filter_rerank_embeddings, gallery_embeddings)


        # then the same thing, but with the two most similar embeddings from gallery_embeddings

        sorted_distances, sorted_indices = torch.sort(distances, dim=1)
        first_gallery_idx = class_ranks[:, 0]
        first_gallery_dstx = sorted_distances[:, 0]
        second_gallery_idx = class_ranks[:, 1]
        second_gallery_dstx = sorted_distances[:, 1]

        rerank_embeddings1 = gallery_embeddings.index_select(0, first_gallery_idx)
        rerank_embeddings2 = gallery_embeddings.index_select(0, second_gallery_idx)

        mask1 = first_gallery_dstx < 0.8
        mask2 = second_gallery_dstx < 0.8

        filter_rerank_embeddings1 = torch.where(
            mask1.view(-1, 1), rerank_embeddings1, query_embeddings
        )
        filter_rerank_embeddings2 = torch.where(
            mask2.view(-1, 1), rerank_embeddings2, query_embeddings
        )

        filter_rerank_embeddings = (
            0.3 * filter_rerank_embeddings1
            + 0.3 * filter_rerank_embeddings2
            + 0.4 * query_embeddings
        )

        distances = torch.cdist(filter_rerank_embeddings, gallery_embeddings)

        sorted_distances = torch.argsort(distances, dim=1)
        sorted_distances = sorted_distances.cpu().numpy()[:, :1000]
        class_ranks = sorted_distances
        return class_ranks
