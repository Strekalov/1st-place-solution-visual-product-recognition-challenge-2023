import torch
from torch.utils.data.sampler import Sampler
import numpy as np
import collections


NUMPY_RANDOM = np.random

def get_labels_to_indices(labels, is_galleries):
    """
    Creates labels_to_indices, which is a dictionary mapping each label
    to a numpy array of indices that will be used to index into self.dataset
    """
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    if torch.is_tensor(is_galleries):
        is_galleries = is_galleries.cpu().numpy()

    labels_to_indices = collections.defaultdict(list)
    labels_to_mask = collections.defaultdict(list)

    for i, label in enumerate(labels):
        is_gallery = is_galleries[i]
        labels_to_indices[label].append(i)
        labels_to_mask[label].append(is_gallery)

    for k, v in labels_to_indices.items():
        labels_to_indices[k] = np.array(v, dtype=int)

    for k, v in labels_to_mask.items():
        labels_to_mask[k] = np.array(v, dtype=int)
    return labels_to_indices, labels_to_mask




def safe_random_choice(input_data, mask, size):
    """
    Randomly samples without replacement from a sequence. It is "safe" because
    if len(input_data) < size, it will randomly sample WITH replacement
    Args:
        input_data is a sequence, like a torch tensor, numpy array,
                        python list, tuple etc
        size is the number of elements to randomly sample from input_data
    Returns:
        An array of size "size", randomly sampled from input_data
    """
    # Делим данные на две части. От галереи и от отзывов
    gallery_data = np.ma.masked_array(input_data, mask).compressed()
    comment_data = np.ma.masked_array(input_data, np.logical_not(mask)).compressed()

    # Обрабатываем случаи, когда фоток из галереи или из отзывов нет
    if len(gallery_data) == 0:
        replace = len(comment_data) < size
        return NUMPY_RANDOM.choice(comment_data, size=size, replace=replace)
    if len(comment_data) == 0:
        replace = len(gallery_data) < size
        return NUMPY_RANDOM.choice(gallery_data, size=size, replace=replace)

    # Если требуется нечетное число семплев, 
    # то смотрим в галереи больше фоток или в отзывах.
    #  Где больше, там и на один семпл в батче будет больше.
    # Если одинаково, то берем из отзывов на один семпл в батче больше
    if len(gallery_data) > len(comment_data):
        gallery_size = (size // 2) + (size % 2)
        comment_size = size - gallery_size
    else:
        gallery_size = size // 2
        comment_size = size - gallery_size
    # Если фоток меньше требуемого размера, то составляем семплы с повторением
    gallery_replace = len(gallery_data) < gallery_size
    comment_replace = len(comment_data) < comment_size

    gallery_result = NUMPY_RANDOM.choice(gallery_data, size=gallery_size, replace=gallery_replace)
    comment_result = NUMPY_RANDOM.choice(comment_data, size=comment_size, replace=comment_replace)
    return np.concatenate((gallery_result, comment_result), axis=0)


# modified from
# https://raw.githubusercontent.com/bnulihaixia/Deep_metric/master/utils/sampler.py
class MPerClassBalanceSampler(Sampler):
    """
    At every iteration, this will return m samples per class. For example,
    if dataloader's batchsize is 100, and m = 5, then 20 classes with 5 samples
    each will be returned
    """

    def __init__(self, labels, is_galleries, m, batch_size=None, length_before_new_iter=100000):
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        self.m_per_class = int(m)
        self.batch_size = int(batch_size) if batch_size is not None else batch_size
        self.labels_to_indices, self.labels_to_mask = get_labels_to_indices(labels, is_galleries)
        self.labels = list(self.labels_to_indices.keys())
        self.length_of_single_pass = self.m_per_class * len(self.labels)
        self.list_size = length_before_new_iter
        if self.batch_size is None:
            if self.length_of_single_pass < self.list_size:
                self.list_size -= (self.list_size) % (self.length_of_single_pass)
        else:
            assert self.list_size >= self.batch_size
            assert (
                self.length_of_single_pass >= self.batch_size
            ), "m * (number of unique labels) must be >= batch_size"
            assert (
                self.batch_size % self.m_per_class
            ) == 0, "m_per_class must divide batch_size without any remainder"
            self.list_size -= self.list_size % self.batch_size

    def __len__(self):
        return self.list_size

    def __iter__(self):
        idx_list = [0] * self.list_size
        i = 0
        num_iters = self.calculate_num_iters()
        for _ in range(num_iters):
            NUMPY_RANDOM.shuffle(self.labels)
            if self.batch_size is None:
                curr_label_set = self.labels
            else:
                curr_label_set = self.labels[: self.batch_size // self.m_per_class]
            for label in curr_label_set:
                t = self.labels_to_indices[label]
                mask = self.labels_to_mask[label]
                idx_list[i : i + self.m_per_class] = safe_random_choice(
                    t, mask, size=self.m_per_class
                )
                i += self.m_per_class
        return iter(idx_list)

    def calculate_num_iters(self):
        divisor = (
            self.length_of_single_pass if self.batch_size is None else self.batch_size
        )
        return self.list_size // divisor if divisor < self.list_size else 1
# class MPerClassBalanceSampler(Sampler):
#     """
#     At every iteration, this will return m samples per class. For example,
#     if dataloader's batchsize is 100, and m = 5, then 20 classes with 5 samples
#     each will be returned
#     """

#     def __init__(self, labels, is_galleries, m, batch_size=None, length_before_new_iter=100000):
#         if isinstance(labels, torch.Tensor):
#             labels = labels.numpy()
#         self.m_per_class = int(m)
#         self.batch_size = int(batch_size) if batch_size is not None else batch_size
#         self.labels_to_indices, self.labels_to_mask = get_labels_to_indices(labels, is_galleries)
#         self.uniq_labels = list(self.labels_to_indices.keys())
#         self.length_of_single_pass = self.m_per_class * len(self.uniq_labels)
#         self.list_size = length_before_new_iter
#         self.labels_per_iter = self.batch_size // self.m_per_class
#         if self.batch_size is None:
#             if self.length_of_single_pass < self.list_size:
#                 self.list_size -= (self.list_size) % (self.length_of_single_pass)
#         else:
#             assert self.list_size >= self.batch_size
#             assert (
#                 self.length_of_single_pass >= self.batch_size
#             ), "m * (number of unique labels) must be >= batch_size"
#             assert (
#                 self.batch_size % self.m_per_class
#             ) == 0, "m_per_class must divide batch_size without any remainder"
#             self.list_size -= self.list_size % self.batch_size

#     def __len__(self):
#         return self.list_size

#     def __iter__(self):
#         idx_list = [0] * self.list_size
#         i = 0
#         num_iters = self.calculate_num_iters()
#         NUMPY_RANDOM.shuffle(self.uniq_labels)
        
#         for j in range(num_iters):
            
#             if self.batch_size is None:
#                 curr_label_set = self.uniq_labels
#             else:
#                 curr_label_set = self.uniq_labels[j * self.labels_per_iter:(j + 1) * self.labels_per_iter]
#             for label in curr_label_set:
#                 t = self.labels_to_indices[label]
#                 mask = self.labels_to_mask[label]
#                 idx_list[i : i + self.m_per_class] = safe_random_choice(
#                     t, mask, size=self.m_per_class
#                 )
#                 i += self.m_per_class
#         return iter(idx_list)

#     def calculate_num_iters(self):
#         divisor = (
#             self.length_of_single_pass if self.batch_size is None else self.batch_size
#         )
#         # return self.list_size // divisor if divisor < self.list_size else 1
#         return len(self.uniq_labels) // self.labels_per_iter

if __name__ == '__main__':
    labels = np.array([1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,4,4,4,4,4])
    is_galleries = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0])
    print(len(labels), len(is_galleries), end='\n\n')
    #  Тот же семплер. Только нужно добавить маску из галереи или нет
    sampler = MPerClassSampler(
        labels, is_galleries, m=7, batch_size=21, length_before_new_iter=len(labels)
        )
    for elem in sampler:
        print(elem, labels[elem], is_galleries[elem])
    print()
    print(len(sampler))
    # for i in range(len(labels)):
    #     print(labels[i], is_galleries[i])

