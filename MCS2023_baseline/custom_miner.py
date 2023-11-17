import torch

from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.miners.base_miner import BaseMiner


def get_matches_and_diffs(labels, ref_labels=None):
    if ref_labels is None:
        ref_labels = labels
    labels1 = labels.unsqueeze(1)
    labels2 = ref_labels.unsqueeze(0)

    matches = (labels1 == labels2).byte()
    diffs = matches ^ 1
    if ref_labels is labels:
        matches.fill_diagonal_(0)
    return matches, diffs


def get_all_pairs_indices(labels, ref_labels=None, mode='qg'):
    """
    Given a tensor of labels, this will return 4 tensors.
    The first 2 tensors are the indices which form all positive pairs
    The second 2 tensors are the indices which form all negative pairs
    """
    # gallery_data = torch.where(ref_labels.bool(), labels, -1)
    # comment_data = torch.where(torch.logical_not(ref_labels), labels, -2)
    # gallery_data = torch.masked_select(labels, ref_labels.bool())
    # comment_data = torch.masked_select(labels, torch.logical_not(ref_labels))
    # №№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№
    # Как сделать быстрее. 
    
    gallery_data = ref_labels.bool() == True
    querry_data = ref_labels.bool() == False
    # print(gallery_data.shape, querry_data.shape)

    matches, diffs = get_matches_and_diffs(labels, None)
    a1_idx, p_idx = torch.where(matches)
    a2_idx, n_idx = torch.where(diffs)
    # print(matches.shape, diffs.shape)
    # print(a1_idx.shape, p_idx.shape)
    # print(a2_idx.shape, n_idx.shape)

    # print(p_idx.shape)
    # print(gallery_data.shape)
    # print(gallery_data[p_idx].shape)
    # print(p_idx[gallery_data[p_idx]].shape)
    
    
    p_idx_mask = gallery_data[p_idx] #torch.tensor([x in gallery_data for x in p_idx]).to('cuda:0')*1
    n_idx_mask = gallery_data[n_idx] # torch.tensor([x in gallery_data for x in n_idx]).to('cuda:0')*1

    if mode == 'q':
        a1_idx_mask = querry_data[a1_idx] #torch.tensor([x in querry_data for x in a1_idx]).to('cuda:0')*1
        pair1_mask = torch.logical_and(a1_idx_mask, p_idx_mask) * 1

        a2_idx_mask = querry_data[a2_idx]  #torch.tensor([x in querry_data for x in a2_idx]).to('cuda:0')*1
        pair2_mask = torch.logical_and(a2_idx_mask, n_idx_mask) * 1
        
    else:
        pair1_mask = p_idx_mask
        pair2_mask = n_idx_mask
    # №№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№
    # gallery_data = (ref_labels.bool() == True).nonzero(as_tuple=True)[0]
    # querry_data = (ref_labels.bool() == False).nonzero(as_tuple=True)[0]

    # matches, diffs = get_matches_and_diffs(labels, None)
    # a1_idx, p_idx = torch.where(matches)
    # a2_idx, n_idx = torch.where(diffs)
    
    # p_idx_mask = torch.tensor([x in gallery_data for x in p_idx]).to('cuda:0')*1
    # n_idx_mask = torch.tensor([x in gallery_data for x in n_idx]).to('cuda:0')*1

    # if mode == 'q':
    #     a1_idx_mask = torch.tensor([x in querry_data for x in a1_idx]).to('cuda:0')*1
    #     pair1_mask = torch.logical_and(a1_idx_mask, p_idx_mask) * 1

    #     a2_idx_mask = torch.tensor([x in querry_data for x in a2_idx]).to('cuda:0')*1
    #     pair2_mask = torch.logical_and(a2_idx_mask, n_idx_mask) * 1
        
    # else:
    #     pair1_mask = p_idx_mask
    #     pair2_mask = n_idx_mask
    # for a1, m1, p1, c1 in zip(a1_idx, a1_idx_mask, p_idx_mask, pair1_mask):
    #     print(a1.item(), m1.item(), p1.item(), c1.item())
    
    # for a1, m1, p1, c1 in zip(a2_idx, a2_idx_mask, n_idx_mask, pair2_mask):
    #     print(a1.item(), m1.item(), p1.item(), c1.item())
    a1_idx = torch.masked_select(a1_idx, pair1_mask.bool())
    a2_idx = torch.masked_select(a2_idx, pair2_mask.bool())
    p_idx = torch.masked_select(p_idx, pair1_mask.bool())
    n_idx = torch.masked_select(n_idx, pair2_mask.bool())
    # print(labels.device)
    # print(ref_labels.device)
    # print(gallery_data.device)
    # print(matches.device)
    # print(p_idx_mask.device)

    return a1_idx, p_idx, a2_idx, n_idx


class BatchEasyHardMinerCustom(BaseMiner):
    HARD = "hard"
    SEMIHARD = "semihard"
    EASY = "easy"
    ALL = "all"
    all_batch_mining_strategies = [HARD, SEMIHARD, EASY, ALL]

    def __init__(
        self,
        pos_strategy=EASY,
        neg_strategy=SEMIHARD,
        allowed_pos_range=None,
        allowed_neg_range=None,
        mode='q',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.mode = mode
        if not (
            pos_strategy in self.all_batch_mining_strategies
            and neg_strategy in self.all_batch_mining_strategies
        ):
            raise ValueError(
                '\npos_strategy must be one of "{0}"\nneg_strategy must be one of "{0}"'.format(
                    '" or "'.join(self.all_batch_mining_strategies)
                )
            )
        if pos_strategy == neg_strategy == self.SEMIHARD:
            raise ValueError('pos_strategy and neg_strategy cannot both be "semihard"')

        if (pos_strategy == self.ALL and neg_strategy == self.SEMIHARD) or (
            neg_strategy == self.ALL and pos_strategy == self.SEMIHARD
        ):
            raise ValueError('"semihard" cannot be used in combination with "all"')

        self.pos_strategy = pos_strategy
        self.neg_strategy = neg_strategy
        self.allowed_pos_range = allowed_pos_range
        self.allowed_neg_range = allowed_neg_range

        self.add_to_recordable_attributes(
            list_of_names=[
                "easiest_triplet",
                "hardest_triplet",
                "easiest_pos_pair",
                "hardest_pos_pair",
                "easiest_neg_pair",
                "hardest_neg_pair",
            ],
            is_stat=True,
        )

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        mat = self.distance(embeddings, ref_emb)
        a1_idx, p_idx, a2_idx, n_idx = get_all_pairs_indices(labels, ref_labels, self.mode)
        a = torch.arange(mat.size(0), device=mat.device)
        ref_emb = None
        ref_labels = None

        if self.pos_strategy == self.SEMIHARD and self.neg_strategy != self.ALL:
            (negative_dists, negative_indices), a2n_keep = self.get_negatives(
                mat, a2_idx, n_idx
            )
            (positive_dists, positive_indices), a1p_keep = self.get_positives(
                mat, a1_idx, p_idx, negative_dists
            )
        elif self.neg_strategy == self.SEMIHARD and self.pos_strategy != self.ALL:
            (positive_dists, positive_indices), a1p_keep = self.get_positives(
                mat, a1_idx, p_idx
            )
            (negative_dists, negative_indices), a2n_keep = self.get_negatives(
                mat, a2_idx, n_idx, positive_dists
            )
        else:
            if self.pos_strategy != self.ALL:
                (positive_dists, positive_indices), a1p_keep = self.get_positives(
                    mat, a1_idx, p_idx
                )
            if self.neg_strategy != self.ALL:
                (negative_dists, negative_indices), a2n_keep = self.get_negatives(
                    mat, a2_idx, n_idx
                )

        if self.ALL not in [self.pos_strategy, self.neg_strategy]:
            a_keep_idx = torch.where(a1p_keep & a2n_keep)
            self.set_stats(positive_dists[a_keep_idx], negative_dists[a_keep_idx])
            a = a[a_keep_idx]
            p = positive_indices[a_keep_idx]
            n = negative_indices[a_keep_idx]
            return a, p, a, n
        elif self.pos_strategy == self.ALL and self.neg_strategy != self.ALL:
            self.set_stats(mat[a1_idx, p_idx], negative_dists[a2n_keep])
            a2 = a[a2n_keep]
            n = negative_indices[a2n_keep]
            return a1_idx, p_idx, a2, n
        elif self.pos_strategy != self.ALL and self.neg_strategy == self.ALL:
            self.set_stats(positive_dists[a1p_keep], mat[a2_idx, n_idx])
            a1 = a[a1p_keep]
            p = positive_indices[a1p_keep]
            return a1, p, a2_idx, n_idx
        else:
            self.set_stats(mat[a1_idx, p_idx], mat[a2_idx, n_idx])
            return a1_idx, p_idx, a2_idx, n_idx

    def get_positives(self, mat, a1_idx, p_idx, negative_dists=None):
        pos_func = self.get_mine_function(self.pos_strategy)
        return pos_func(mat, a1_idx, p_idx, self.allowed_pos_range, negative_dists)

    def get_negatives(self, mat, a2_idx, n_idx, positive_dists=None):
        neg_func = self.get_mine_function(
            self.EASY if self.neg_strategy in [self.HARD, self.SEMIHARD] else self.HARD
        )
        return neg_func(mat, a2_idx, n_idx, self.allowed_neg_range, positive_dists)

    def get_mine_function(self, strategy):
        if strategy in [self.HARD, self.SEMIHARD]:
            mine_func = (
                self.get_min_per_row
                if self.distance.is_inverted
                else self.get_max_per_row
            )
        elif strategy == self.EASY:
            mine_func = (
                self.get_max_per_row
                if self.distance.is_inverted
                else self.get_min_per_row
            )
        else:
            raise NotImplementedError

        return mine_func

    def get_max_per_row(self, *args, **kwargs):
        return self.get_x_per_row("max", *args, **kwargs)

    def get_min_per_row(self, *args, **kwargs):
        return self.get_x_per_row("min", *args, **kwargs)

    def get_x_per_row(
        self,
        xtype,
        mat,
        anchor_idx,
        other_idx,
        val_range=None,
        semihard_thresholds=None,
    ):
        assert xtype in ["min", "max"]
        inf = c_f.pos_inf(mat.dtype) if xtype == "min" else c_f.neg_inf(mat.dtype)
        mask = torch.ones_like(mat) * inf
        mask[anchor_idx, other_idx] = 1
        if semihard_thresholds is not None:
            if xtype == "min":
                condition = mat <= semihard_thresholds.unsqueeze(1)
            else:
                condition = mat >= semihard_thresholds.unsqueeze(1)
            mask[condition] = inf
        if val_range is not None:
            mask[(mat > val_range[1]) | (mat < val_range[0])] = inf

        non_inf_rows = torch.any(mask != inf, dim=1)
        mat = mat.clone()
        mat[mask == inf] = inf
        dist_fn = torch.min if xtype == "min" else torch.max
        return dist_fn(mat, dim=1), non_inf_rows

    def set_stats(self, positive_dists, negative_dists):
        if self.collect_stats:
            with torch.no_grad():
                len_pd = len(positive_dists)
                len_pn = len(negative_dists)
                if (
                    len_pd > 0
                    and len_pn > 0
                    and self.ALL not in [self.pos_strategy, self.neg_strategy]
                ):
                    easiest_triplet_func = self.get_func_for_stats(False)
                    hardest_triplet_func = self.get_func_for_stats(True)
                    self.easiest_triplet = easiest_triplet_func(
                        positive_dists - negative_dists
                    ).item()
                    self.hardest_triplet = hardest_triplet_func(
                        positive_dists - negative_dists
                    ).item()
                if len_pd > 0:
                    easy_pos_func = self.get_func_for_stats(False)
                    hard_pos_func = self.get_func_for_stats(True)
                    self.easiest_pos_pair = easy_pos_func(positive_dists).item()
                    self.hardest_pos_pair = hard_pos_func(positive_dists).item()
                if len_pn > 0:
                    easy_neg_func = self.get_func_for_stats(True)
                    hard_neg_func = self.get_func_for_stats(False)
                    self.easiest_neg_pair = easy_neg_func(negative_dists).item()
                    self.hardest_neg_pair = hard_neg_func(negative_dists).item()

    def get_func_for_stats(self, min_if_inverted):
        if min_if_inverted:
            return torch.min if self.distance.is_inverted else torch.max
        return torch.max if self.distance.is_inverted else torch.min



