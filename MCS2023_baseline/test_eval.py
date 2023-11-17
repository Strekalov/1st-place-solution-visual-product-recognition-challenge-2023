import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
import pandas as pd
from mean_average_precision import calculate_map
from data_utils import dataset

def db_augmentation(query_vecs, reference_vecs, top_k=3):
    """
    Database-side feature augmentation (DBA)
    Albert Gordo, et al. "End-to-end Learning of Deep Visual Representations for Image Retrieval,"
    International Journal of Computer Vision. 2017.
    https://link.springer.com/article/10.1007/s11263-017-1016-8
    """
    weights = torch.logspace(0, -2., top_k+1).cuda()

    # Query augmentation
    sim_mat = torch.cdist(query_vecs, reference_vecs)

    indices = torch.argsort(sim_mat, dim=1)

    top_k_ref = reference_vecs[indices[:, :top_k], :]
    query_vecs = torch.tensordot(weights, torch.cat([torch.unsqueeze(query_vecs, 1), top_k_ref], dim=1), dims=([0], [1]))

    # Reference augmentation
    sim_mat = torch.cdist(reference_vecs, reference_vecs)
    indices = torch.argsort(sim_mat, dim=1)

    top_k_ref = reference_vecs[indices[:, :top_k+1], :]
    reference_vecs = torch.tensordot(weights, top_k_ref, dims=([0], [1]))
    # reference_vecs = torch.tensordot(weights, torch.cat([torch.unsqueeze(query_vecs, 1), top_k_ref], dim=1), dims=([0], [1]))

    return query_vecs, reference_vecs

def get_wb_val_dataloaders():
    query_dataset = dataset.WBValDataset(root="/mnt/wb_products_dataset",
        annotation_file="/mnt/wb_products_dataset/testrerank2.csv", mode="query")
    
    gallery_dataset = dataset.WBValDataset(root="/mnt/wb_products_dataset",
        annotation_file="/mnt/wb_products_dataset/testrerank2.csv", mode="gallery")
    print("wb gallery, query", len(gallery_dataset), len(query_dataset))
    query_loader = torch.utils.data.DataLoader(
        query_dataset,
        batch_size=128,
        shuffle=False,
        pin_memory=True,
        num_workers=30,
    )
        
    gallery_loader = torch.utils.data.DataLoader(
        gallery_dataset,
        batch_size=128,
        shuffle=False,
        pin_memory=True,
        num_workers=30,
    )

    return gallery_loader, query_loader
    

def validation_wb(
    # model: torch.nn.Module,
    gallery_loader: torch.utils.data.DataLoader,
    query_loader: torch.utils.data.DataLoader,
) -> None:
    """
    Model validation function for one epoch
    :param model: model architecture
    :param val_loader: dataloader for batch generation
    :param criterion: selected criterion for calculating the loss function
    :param epoch (int): epoch number
    :return: float: avg acc
    """

    print("Calculating embeddings")
    # 111349 193205 28620 45408
    gallery_embeddings = torch.zeros((9066, 2048), device="cuda", requires_grad=False, dtype=torch.half)
    query_embeddings = torch.zeros((9066, 2048), device="cuda", requires_grad=False, dtype=torch.half)
    # gallery_embeddings = np.zeros((50759, config.model.embedding_dim))
    # query_embeddings = np.zeros((50759, config.model.embedding_dim))
    gallery_product_ids = np.zeros(9066).astype(np.int32)
    query_product_ids = np.zeros(9066).astype(np.int32)
    # gallery_product_ids = 
    # query_product_ids = torch.zeros((50759),)
    # print(len(query_loader), "228322")
    # model.eval()
    with torch.no_grad():
        for i, (outputs, targets) in tqdm(enumerate(gallery_loader), total=len(gallery_loader)):
            # images = images.contiguous(memory_format=torch.channels_last)
            outputs = outputs.half().cuda()
            # outputs = model(images.half().cuda())
            # outputs = outputs[2]
            # outputs = outputs.data.cpu().numpy()
            # print(outputs.shape, "228322!!!")
            gallery_embeddings[
                i
                * 128 : (
                    i * 128 + 128
                ),
                :,
            ] = outputs
            gallery_product_ids[
                i
                * 128 : (
                    i * 128 + 128
                )
            ] = targets

        for i, (outputs, targets) in tqdm(enumerate(query_loader), total=len(query_loader)):
            # images = images.contiguous(memory_format=torch.channels_last)
            # outputs = model(images.half().cuda())
            outputs = outputs.half().cuda()
            # outputs = outputs[2]

            # outputs = outputs.data.cpu().numpy()
            query_embeddings[
                i
                * 128 : (
                    i * 128 + 128
                ),
                :,
            ] = outputs
            
            query_product_ids[
                i
                * 128 : (
                    i * 128 + 128
                )
            ] = targets

        
        
        query_embeddings, gallery_embeddings = query_embeddings.float(), gallery_embeddings.float()
        concat = torch.cat((query_embeddings, gallery_embeddings), dim=0)
        center = torch.mean(concat, dim=0)
        query_embeddings = query_embeddings-center
        gallery_embeddings = gallery_embeddings-center
        gallery_embeddings = torch.nn.functional.normalize(gallery_embeddings, p=2.0, dim=1)
        query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2.0, dim=1)

        query_embeddings, gallery_embeddings = db_augmentation(query_embeddings, gallery_embeddings, top_k=6)
        # query_embeddings, gallery_embeddings = average_query_expansion(query_embeddings, gallery_embeddings, top_k=2)
        distances = torch.cdist(query_embeddings, gallery_embeddings)
        # distances = distances.cpu().numpy()
        # sorted_distances = torch.argsort(distances, dim=1)
        sorted_distances, sorted_indices = torch.sort(distances, dim=1)
        


        
        # sorted_distances = torch.argsort(distances, dim=1)
        # sorted_distances = sorted_distances.cpu().numpy()[:, :1000]
        # # distances = pairwise_distances(query_embeddings, gallery_embeddings, metric="cosine", n_jobs=4)
        # # print(distances.shape)
        # # try:    
        # #     sorted_distances = np.argsort(distances, axis=1)[:, :1000]
        # # except Exception as ex:
        # #     print(ex)
        # #     sorted_distances = np.argsort(distances, axis=1)[:, :1000]
        class_ranks = sorted_indices
        # print(len(class_ranks))
        
        # first_gallery_distance = sorted_distances[:, 0]
        # first_gallery_idx = sorted_indices[:, 0]
        # mask = gallery_distances < 0.25

        # torch.where(query_similarities.max(dim=1).values < threshold)[0]
        

        first_gallery_idx = class_ranks[:, 0]
        first_gallery_dstx = sorted_distances[:, 0]
        # second_gallery_idx = class_ranks[:, 1]
        # second_gallery_dstx = sorted_distances[:, 1]
        # third_gallery_idx = class_ranks[:, 2]
        # third_gallery_dstx = sorted_distances[:, 2]
        # rerank_embeddings = torch.take_along_dim(gallery_embeddings, first_gallery_idx, dim=0)
        rerank_embeddings1 = gallery_embeddings.index_select(0, first_gallery_idx)
        # rerank_embeddings2 = gallery_embeddings.index_select(0, second_gallery_idx)
        # rerank_embeddings3 = gallery_embeddings.index_select(0, third_gallery_idx)
        # distance = torch.cdist(query_embeddings, rerank_embeddings)
        # similarities = torch.mm(query_embeddings, rerank_embeddings.t())
        # mask = first_gallery_dstx < 0.8
        mask1 = first_gallery_dstx < 0.8322
        # mask2 = second_gallery_dstx < 0.8
        # mask3 = third_gallery_dstx < 0.6
        
        # mask = similarities < 0.4

        # объединяем векторы по условию
        # print(distance)
        # filter_rerank_embeddings = torch.where(distance > 0.7, query_embeddings, rerank_embeddings)
        
        # for i in range(rerank_embeddings.shape[0]):
            
        
        filter_rerank_embeddings1 = torch.where(mask1.view(-1, 1), rerank_embeddings1, query_embeddings)
        # filter_rerank_embeddings2 = torch.where(mask2.view(-1, 1), rerank_embeddings2, query_embeddings)
        # filter_rerank_embeddings3 = torch.where(mask3.view(-1, 1), rerank_embeddings3, query_embeddings)
        # filter_rerank_embeddings = 0.4*filter_rerank_embeddings1+0.2*filter_rerank_embeddings2+ 0.4*query_embeddings
        # filter_rerank_embeddings = query_embeddings
        filter_rerank_embeddings = 0.5*filter_rerank_embeddings1 + 0.5*query_embeddings
        distances = torch.cdist(filter_rerank_embeddings, gallery_embeddings)
        # sorted_distances = torch.argsort(distances, dim=1)
        # sorted_distances = sorted_distances.cpu().numpy()[:, :1000]

        # second_gallery_idx = class_ranks[:, 1]
        # rerank_embeddings = np.take(gallery_embeddings, first_gallery_idx, axis=0)
        

        
        # rerank_embeddings = (rerank_embeddings+query_embeddings)/2
        
        # distances = pairwise_distances(rerank_embeddings, gallery_embeddings)

        # try:
        #     sorted_distances = np.argsort(distances, axis=1)[:, :1000]
        # except Exception as ex:
        #     print(ex)
        #     sorted_distances = np.argsort(distances, axis=1)
        # first_gallery_idx = np.expand_dims(first_gallery_idx, axis=0)

        # class_ranks = sorted_distances
        sorted_distances, sorted_indices = torch.sort(distances, dim=1)
        first_gallery_idx = class_ranks[:, 0]
        first_gallery_dstx = sorted_distances[:, 0]
        second_gallery_idx = class_ranks[:, 1]
        second_gallery_dstx = sorted_distances[:, 1]
        # third_gallery_idx = class_ranks[:, 2]
        # third_gallery_dstx = sorted_distances[:, 2]
        # rerank_embeddings = torch.take_along_dim(gallery_embeddings, first_gallery_idx, dim=0)
        rerank_embeddings1 = gallery_embeddings.index_select(0, first_gallery_idx)
        rerank_embeddings2 = gallery_embeddings.index_select(0, second_gallery_idx)
        # rerank_embeddings3 = gallery_embeddings.index_select(0, third_gallery_idx)
        # distance = torch.cdist(query_embeddings, rerank_embeddings)
        # similarities = torch.mm(query_embeddings, rerank_embeddings.t())
        # mask = first_gallery_dstx < 0.8
        mask1 = first_gallery_dstx < 0.8322
        mask2 = second_gallery_dstx < 0.8322
        # mask3 = third_gallery_dstx < 0.6
        
        # mask = similarities < 0.4

        # объединяем векторы по условию
        # print(distance)
        # filter_rerank_embeddings = torch.where(distance > 0.7, query_embeddings, rerank_embeddings)
        
        # for i in range(rerank_embeddings.shape[0]):
            
        
        filter_rerank_embeddings1 = torch.where(mask1.view(-1, 1), rerank_embeddings1, query_embeddings)
        filter_rerank_embeddings2 = torch.where(mask2.view(-1, 1), rerank_embeddings2, query_embeddings)
        # filter_rerank_embeddings3 = torch.where(mask3.view(-1, 1), rerank_embeddings3, query_embeddings)
        filter_rerank_embeddings = 0.275*filter_rerank_embeddings1+0.275*filter_rerank_embeddings2+ 0.45*query_embeddings
        # filter_rerank_embeddings = (filter_rerank_embeddings1+filter_rerank_embeddings2+query_embeddings) / 3
        # filter_rerank_embeddings = query_embeddings
        distances = torch.cdist(filter_rerank_embeddings, gallery_embeddings)

        # sorted_distances = torch.argsort(distances, dim=1)
        # sorted_distances = sorted_distances.cpu().numpy()[:, :1000]
        
        # distances = torch.cdist(filter_rerank_embeddings, gallery_embeddings)
        # sorted_distances = torch.argsort(distances, dim=1)
        # sorted_distances = sorted_distances.cpu().numpy()[:, :1000]

        # second_gallery_idx = class_ranks[:, 1]
        # rerank_embeddings = np.take(gallery_embeddings, first_gallery_idx, axis=0)
        

        
        # rerank_embeddings = (rerank_embeddings+query_embeddings)/2
        
        # distances = pairwise_distances(rerank_embeddings, gallery_embeddings)

        # try:
        #     sorted_distances = np.argsort(distances, axis=1)[:, :1000]
        # except Exception as ex:
        #     print(ex)
        #     sorted_distances = np.argsort(distances, axis=1)
        # first_gallery_idx = np.expand_dims(first_gallery_idx, axis=0)

        # class_ranks = sorted_distances
        
        
        #<3 RERANK>
        
        # sorted_distances, sorted_indices = torch.sort(distances, dim=1)
        # first_gallery_idx = class_ranks[:, 0]
        # first_gallery_dstx = sorted_distances[:, 0]
        # second_gallery_idx = class_ranks[:, 1]
        # second_gallery_dstx = sorted_distances[:, 1]
        # third_gallery_idx = class_ranks[:, 2]
        # third_gallery_dstx = sorted_distances[:, 2]
        # # rerank_embeddings = torch.take_along_dim(gallery_embeddings, first_gallery_idx, dim=0)
        # rerank_embeddings1 = gallery_embeddings.index_select(0, first_gallery_idx)
        # rerank_embeddings2 = gallery_embeddings.index_select(0, second_gallery_idx)
        # rerank_embeddings3 = gallery_embeddings.index_select(0, third_gallery_idx)
        # # distance = torch.cdist(query_embeddings, rerank_embeddings)
        # # similarities = torch.mm(query_embeddings, rerank_embeddings.t())
        # # mask = first_gallery_dstx < 0.8
        # mask1 = first_gallery_dstx < 0.8
        # mask2 = second_gallery_dstx < 0.8
        # mask3 = third_gallery_dstx < 0.8
        
        # # mask = similarities < 0.4

        # # объединяем векторы по условию
        # # print(distance)
        # # filter_rerank_embeddings = torch.where(distance > 0.7, query_embeddings, rerank_embeddings)
        
        # # for i in range(rerank_embeddings.shape[0]):
            
        
        # filter_rerank_embeddings1 = torch.where(mask1.view(-1, 1), rerank_embeddings1, query_embeddings)
        # filter_rerank_embeddings2 = torch.where(mask2.view(-1, 1), rerank_embeddings2, query_embeddings)
        # filter_rerank_embeddings3 = torch.where(mask3.view(-1, 1), rerank_embeddings3, query_embeddings)
        # filter_rerank_embeddings = 0.3*filter_rerank_embeddings1+0.3*filter_rerank_embeddings2+ 0.05*filter_rerank_embeddings3+0.4*query_embeddings
        # # filter_rerank_embeddings = (filter_rerank_embeddings1+filter_rerank_embeddings2+query_embeddings) / 3
        # # filter_rerank_embeddings = query_embeddings
        # distances = torch.cdist(filter_rerank_embeddings, gallery_embeddings)
        
        
        
        sorted_distances = torch.argsort(distances, dim=1)
        sorted_distances = sorted_distances.cpu().numpy()[:, :1000]
        class_ranks=sorted_distances
        
        
        
        
        
        public_map = calculate_map(sorted_distances, query_product_ids, gallery_product_ids)
        # print(f"Validation on epoch {epoch}: mAP: {public_map}")
        # with torch.no_grad():
        #     torch.cuda.empty_cache()
        #     gc.collect()
        return public_map
    
    
def main():
    gallery_dataloader, query_dataloader = get_wb_val_dataloaders()
    # model = torch.jit.load("/home/cv_user/visual-product-recognition-2023-giga-flex/MCS2023_baseline/convnext_6398.pt")
    # model.half().cuda()
    # model.eval()
    
    map = validation_wb(gallery_dataloader, query_dataloader)
    print(map)
    
if __name__ == "__main__":
    main()

    