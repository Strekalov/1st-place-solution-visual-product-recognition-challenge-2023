import os
import torch
import cv2
import pandas as pd
import torch.utils.data as data
import torchvision as tv
from PIL import Image
from turbojpeg import TurboJPEG
from .augmentations import get_gallery_transform, get_query_transform, get_eval_gallery_aug, get_eval_query_aug
from torch import nn
import numpy as np
from pathlib import Path

def read_image(image_file):
    img = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img is None:
        raise ValueError("Failed to read {}".format(image_file))
    return img


def set_labels_to_range(labels):
    """
    set the labels so it follows a range per level of semantic
    usefull for example for CSLLoss
    """
    new_labels = []
    print(labels.shape)
    # for lvl in range(labels.shape[1]):
    unique_group = sorted(set(labels[:, 0]))
    print('unique_group', len(unique_group))
    
    conversion = {x: i for i, x in enumerate(unique_group)}
    new_lvl_labels = [conversion[x] for x in labels[:, 0]]
    new_labels.append(new_lvl_labels)

    unique_classes = sorted(set(labels[:, 1]))
    print('unique_classes', len(unique_classes))
    conversion = {x: i for i, x in enumerate(unique_classes)}
    new_lvl_labels = [conversion[x] for x in labels[:, 1]]
    new_labels.append(new_lvl_labels)

    # print(np.stack(new_labels, axis=1).shape)

    return np.stack(new_labels, axis=1)


class WBFlexDataset(data.Dataset):
    def __init__(
        self,
        root,
        annotation_file,
        config
    ):
        self.root = root
        try:
            table = pd.read_csv(annotation_file, delimiter=";")
            # print(table)
            paths = table["img_path"].tolist()
        except:
            table = pd.read_csv(annotation_file, delimiter=",")
            print(table)
            paths = table["img_path"].tolist()
        self.is_galleries = ['gallery' in x for x in paths]
        self.paths = [f"{self.root}{x}" for x in paths]
        labels = table[["category_id", "class_id"]].to_numpy()
        
        self.classes_count = len(table['class_id'].unique())
        print("длина датасета", len(table))
        self.groups_count = len(table['category_id'].unique())
        # print(classes_count, group_count)
        # exit(1)
        type_ids = table[["type_id"]].to_numpy(dtype=np.int32)
        self.labels = set_labels_to_range(labels)
        
        self.labels = np.hstack([self.labels, type_ids])
        # print(self.labels)
        # exit(1)
        # self.labels = self.imlist[["category_id", "class_id"]].to_numpy()

        self.gallery_transform = get_gallery_transform(config)
        self.query_transform = get_query_transform(config)
        self.turbo_jpeg = TurboJPEG()
        # self.feature_extractor = feature_extractor

    def __getitem__(self, index):
        cv2.setNumThreads(120)

        full_imname = self.paths[index]
        category_id, class_id, type_id= self.labels[index]
        is_gallery = self.is_galleries[index]
        # impath, class_id, category_id, type_id = self.imlist.iloc[index]

        # full_imname = os.path.join(self.root, impath)
        # img = read_image(full_imname)
        try:
            with open(file=full_imname, mode="rb") as image_file:
                img = self.turbo_jpeg.decode(image_file.read())
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as ex:
            print(full_imname)
            return torch.zeros(3, 224, 224), category_id, class_id, is_gallery

        transform = self.gallery_transform if type_id == 0 else self.query_transform
        full_imname = Path(full_imname)
        box_txt_path = Path(f"{full_imname.parent}/{full_imname.stem}.txt")
        if type_id == 1 and Path.exists(box_txt_path):

            with box_txt_path.open("r") as file:
                contents = file.readline()
                x1, y1, x2, y2 = contents.split(" ")

                img = img[int(y1):int(y2), int(x1):int(x2), :]
                # with open("images.txt", mode="a") as f:
                #     f.write(str(full_imname) + "/n")
                # cv2.imwrite(f"{full_imname.stem}.jpg", img)
                # exit(1)
        
        img = transform(image=img)["image"]


        return img, category_id, class_id, is_gallery

    def __len__(self):
        return len(self.paths)
    
    
    
class WBValDataset(data.Dataset):
    def __init__(
        self,
        root,
        annotation_file,
        # config,
        mode
    ):
        self.root = root
        try:
            table = pd.read_csv(annotation_file, delimiter=";")
            if mode=="gallery":
                table = table[table["type_id"]==0] 
            else:
                table = table[table["type_id"]==1] 
            # print(table)
            paths = table["img_path"].tolist()
        except:
            table = pd.read_csv(annotation_file, delimiter=",")
            # print(table)
            paths = table["img_path"].tolist()
        self.is_galleries = ['gallery' in x for x in paths]
        self.paths = [f"{self.root}{x}" for x in paths]
        labels = table[["category_id", "class_id"]].to_numpy()
        
        self.classes_count = len(table['class_id'].unique())
        print("длина датасета", len(table))
        self.groups_count = len(table['category_id'].unique())
        # print(classes_count, group_count)
        # exit(1)
        type_ids = table[["type_id"]].to_numpy(dtype=np.int32)
        # self.labels = set_labels_to_range(labels)
        self.labels = labels
        self.labels = np.hstack([self.labels, type_ids])
        # print(self.labels)
        # exit(1)
        # self.labels = self.imlist[["category_id", "class_id"]].to_numpy()
        
        self.gallery_transform = get_eval_gallery_aug()
        self.query_transform = get_eval_query_aug()
        self.turbo_jpeg = TurboJPEG()
        # self.feature_extractor = feature_extractor

    def __getitem__(self, index):
        cv2.setNumThreads(120)

        full_imname = self.paths[index]
        category_id, class_id, type_id= self.labels[index]
        is_gallery = self.is_galleries[index]
        # impath, class_id, category_id, type_id = self.imlist.iloc[index]

        # full_imname = os.path.join(self.root, impath)
        # img = read_image(full_imname)
        # try:
        #     with open(file=full_imname, mode="rb") as image_file:
        #         img = self.turbo_jpeg.decode(image_file.read())
        #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # except Exception as ex:
        #     print(full_imname)
        #     return torch.zeros(3, 224, 224), category_id, class_id, is_gallery

        transform = self.gallery_transform if type_id == 0 else self.query_transform
        full_imname = Path(full_imname)
        box_txt_path = Path(f"{full_imname.parent}/{full_imname.stem}.txt")
        embedding_path = Path(f"{full_imname.parent}/{full_imname.stem}.pt")
        embedding = torch.load(str(embedding_path), map_location="cpu")
        
        
        # if type_id == 1 and Path.exists(box_txt_path):

        #     with box_txt_path.open("r") as file:
        #         contents = file.readline()
        #         x1, y1, x2, y2 = contents.split(" ")

        #         img = img[int(y1):int(y2), int(x1):int(x2), :]
        #         # with open("images.txt", mode="a") as f:
        #         #     f.write(str(full_imname) + "/n")
        #         # cv2.imwrite(f"{full_imname.stem}.jpg", img)
        #         # exit(1)
        
        # img = transform(image=img)["image"]


        return embedding, class_id

    def __len__(self):
        return len(self.paths)

# class Product10KDataset(data.Dataset):
#     def __init__(self, root, annotation_file, transforms, feature_extractor,is_inference=False,
#                  with_bbox=False):
#         self.root = root
#         self.imlist = pd.read_csv(annotation_file)
#         self.imlist.loc[self.imlist["group"] == 360, "group"] = 359

#         self.targets = self.imlist[["group", "class"]].to_numpy()
#         # print(self.targets)
#         # exit(1)
#         # classes_count = len(self.imlist['class'].unique())
#         # grops = self.imlist['group'].unique()
#         # import numpy as np
#         # print(np.sort(grops))
#         # group_count = len(self.imlist['group'].unique())
#         # print(group_count, classes_count, "classes count")
#         # exit(1)
#         self.transforms = transforms
#         self.is_inference = is_inference
#         self.with_bbox = with_bbox
#         self.turbo_jpeg = TurboJPEG()
#         # self.feature_extractor = feature_extractor
#         self.images = []
#         self.groups = []
#         self.targets = []
#         for i in range(len(self.imlist)):
#             impath, target, group = self.imlist.iloc[i]

#             full_imname = os.path.join(self.root, impath)

#             # img = read_image(full_imname)

#             with open(file=full_imname, mode="rb") as image_file:
#                 img = self.turbo_jpeg.decode(image_file.read())
#                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             self.targets.append(target)
#             self.groups.append(group)
#             self.images.append(img)


#     def __getitem__(self, index):
#         cv2.setNumThreads(120)
#         img = self.images[index]
#         group = self.groups[index]
#         target = self.targets[index]

#         img = self.transforms(image=img)["image"]
#         return img, group, target

#         # if self.is_inference:
#         #     impath, _, _ = self.imlist.iloc[index]
#         # else:
#         #     impath, target, group = self.imlist.iloc[index]

#         # full_imname = os.path.join(self.root, impath)
#         # # img = read_image(full_imname)

#         # with open(file=full_imname, mode="rb") as image_file:
#         #     img = self.turbo_jpeg.decode(image_file.read())
#         #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         # # else:
#         # #     data=tv.io.read_file(full_imname)
#         # #     img = tv.io.decode_jpeg(data, device='cuda')


#         # if self.with_bbox:
#         #     x, y, w, h = self.table.loc[index, 'bbox_x':'bbox_h']
#         #     img = img[y:y+h, x:x+w, :]

#         # img = self.transforms(image=img)["image"]


#         #     # img = gpu_transfrom(img)
#         # # img = self.feature_extractor(images=img, return_tensors="pt")
#         # # for k, v in img.items():
#         # #     img[k].squeeze_()  # remove batch dimension
#         # if self.is_inference:
#         #     return img
#         # else:
#         #     # img['labels'] = target
#         #     return img, group, target

#     def __len__(self):
#         return len(self.imlist)


# class SubmissionDataset(data.Dataset):
#     def __init__(self, root, annotation_file, transforms, with_bbox=False):
#         self.root = root
#         self.imlist = pd.read_csv(annotation_file)
#         # cvs=read_csv('sample.csv')
#         # pd.set_option('display.max_rows', 50)
#         # print(self.imlist)
#         # print("len annotation_file ", len(self.imlist))

#         self.transforms = transforms
#         self.with_bbox = with_bbox
#         self.turbo_jpeg = TurboJPEG()

#         self.images = []
#         for i in range(len(self.imlist)):
#             impath, target, group = self.imlist.iloc[i]

#             full_imname = os.path.join(self.root, impath)
#             with open(file=full_imname, mode="rb") as image_file:
#                 img = self.turbo_jpeg.decode(image_file.read())
#                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#             if self.with_bbox:
#                 x, y, w, h = self.imlist.loc[i, 'bbox_x':'bbox_h']
#                 img = img[y:y+h, x:x+w, :]

#             self.images.append(img)

#     def __getitem__(self, index):
#         cv2.setNumThreads(120)
#         img = self.images[index]

#         # img = Image.fromarray(img)
#         # img = self.transforms(img)
#         img = self.transforms(image=img)["image"]
#         return img

#     def __len__(self):
#         return len(self.imlist)


class СdiDataset(data.Dataset):
    def __init__(
        self,
        root,
        annotation_file,
        transforms,
        feature_extractor,
        is_inference=False,
        with_bbox=False,
    ):
        self.root = root
        self.imlist = pd.read_csv(annotation_file)
        # self.imlist.loc[self.imlist["group"] == 360, "group"] = 359

        # self.targets = self.imlist[["group", "class"]].to_numpy()

        # print(self.targets)
        # exit(1)
        # classes_count = len(self.imlist['class'].unique())
        # grops = self.imlist['group'].unique()
        # import numpy as np
        # print(np.sort(grops))
        # group_count = len(self.imlist['category_id'].unique())
        # print(group_count)
        # exit(1)
        # print(group_count, classes_count, "classes count")
        # exit(1)
        self.transforms = transforms
        self.is_inference = is_inference
        self.with_bbox = with_bbox
        self.turbo_jpeg = TurboJPEG()
        # print(self.root)
        # exit(0)
        # self.feature_extractor = feature_extractor

    def __getitem__(self, index):
        cv2.setNumThreads(120)

        if self.is_inference:
            impath, _, _ = self.imlist.iloc[index]
        else:
            impath, group, target = self.imlist.iloc[index]

        full_imname = f"{self.root}/{impath}"
        # img = read_image(full_imname)

        with open(file=full_imname, mode="rb") as image_file:
            img = self.turbo_jpeg.decode(image_file.read())
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # else:
        #     data=tv.io.read_file(full_imname)
        #     img = tv.io.decode_jpeg(data, device='cuda')

        if self.with_bbox:
            x, y, w, h = self.table.loc[index, "bbox_x":"bbox_h"]
            img = img[y : y + h, x : x + w, :]

        img = self.transforms(image=img)["image"]

        # img = gpu_transfrom(img)
        # img = self.feature_extractor(images=img, return_tensors="pt")
        # for k, v in img.items():
        #     img[k].squeeze_()  # remove batch dimension
        if self.is_inference:
            return img
        else:
            # img['labels'] = target
            return img, group, target

    def __len__(self):
        return len(self.imlist)


class DyMLProductDataset(data.Dataset):
    def __init__(
        self,
        root,
        annotation_file,
        transforms,
        feature_extractor,
        is_inference=False,
        with_bbox=False,
    ):
        self.root = root
        table = pd.read_csv(annotation_file)
        self.classes_count = len(table[' fine_id0.jpg'].unique())
        # self.imlist = data[["fname", " fine_id0.jpg", " middle_id"]]
        paths = table["fname"].tolist()
        self.paths = [f"{self.root}/imgs/{x}" for x in paths]
        labels = table[[" middle_id", " fine_id0.jpg"]].to_numpy()
        self.labels = set_labels_to_range(labels)
        self.transforms = transforms
        self.is_inference = is_inference
        self.with_bbox = with_bbox
        self.turbo_jpeg = TurboJPEG()
        # self.feature_extractor = feature_extractor

    def __getitem__(self, index):
        cv2.setNumThreads(120)

        # if self.is_inference:
        #     impath, _, _ = self.imlist.iloc[index]
        # else:
        #     impath, target, group = self.imlist.iloc[index]

        full_imname = self.paths[index]
        group, target = self.labels[index]
        # full_imname= f"{self.root}/imgs/{impath}"
        # img = read_image(full_imname)

        with open(file=full_imname, mode="rb") as image_file:
            img = self.turbo_jpeg.decode(image_file.read())
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # else:
        #     data=tv.io.read_file(full_imname)
        #     img = tv.io.decode_jpeg(data, device='cuda')

        if self.with_bbox:
            x, y, w, h = self.table.loc[index, "bbox_x":"bbox_h"]
            img = img[y : y + h, x : x + w, :]

        img = self.transforms(image=img)["image"]

        # img = gpu_transfrom(img)
        # img = self.feature_extractor(images=img, return_tensors="pt")
        # for k, v in img.items():
        #     img[k].squeeze_()  # remove batch dimension
        if self.is_inference:
            return img
        else:
            # img['labels'] = target
            return img, group, target, 0

    def __len__(self):
        return len(self.paths)


class Product10KDataset(data.Dataset):
    def __init__(
        self,
        root,
        annotation_file,
        transforms,
        feature_extractor,
        is_inference=False,
        with_bbox=False,
    ):
        self.root = root
        self.imlist = pd.read_csv(annotation_file)
        self.imlist.loc[self.imlist["group"] == 360, "group"] = 359

        self.labels = self.imlist[["group", "class"]].to_numpy()
        # print(self.targets)
        # exit(1)
        self.classes_count = len(self.imlist['class'].unique())
        # grops = self.imlist['group'].unique()
        # import numpy as np
        # print(np.sort(grops))
        # group_count = len(self.imlist['group'].unique())
        # print(group_count, classes_count, "classes count")
        # exit(1)
        self.transforms = transforms
        self.is_inference = is_inference
        self.with_bbox = with_bbox
        self.turbo_jpeg = TurboJPEG()
        # self.feature_extractor = feature_extractor

    def __getitem__(self, index):
        cv2.setNumThreads(120)

        if self.is_inference:
            impath, _, _ = self.imlist.iloc[index]
        else:
            impath, target, group = self.imlist.iloc[index]

        full_imname = os.path.join(self.root, impath)
        # img = read_image(full_imname)

        with open(file=full_imname, mode="rb") as image_file:
            img = self.turbo_jpeg.decode(image_file.read())
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # else:
        #     data=tv.io.read_file(full_imname)
        #     img = tv.io.decode_jpeg(data, device='cuda')

        if self.with_bbox:
            x, y, w, h = self.table.loc[index, "bbox_x":"bbox_h"]
            img = img[y : y + h, x : x + w, :]

        img = self.transforms(image=img)["image"]

        # img = gpu_transfrom(img)
        # img = self.feature_extractor(images=img, return_tensors="pt")
        # for k, v in img.items():
        #     img[k].squeeze_()  # remove batch dimension
        if self.is_inference:
            return img
        else:
            # img['labels'] = target
            return img, group, target, 0

    def __len__(self):
        return len(self.imlist)


class SubmissionDataset(data.Dataset):
    def __init__(self, root, annotation_file, transforms, with_bbox=False):
        self.root = root
        self.imlist = pd.read_csv(annotation_file)
        # cvs=read_csv('sample.csv')
        # pd.set_option('display.max_rows', 50)
        # print(self.imlist)
        # print("len annotation_file ", len(self.imlist))

        self.transforms = transforms
        self.with_bbox = with_bbox
        self.turbo_jpeg = TurboJPEG()

    def __getitem__(self, index):
        cv2.setNumThreads(120)

        full_imname = os.path.join(self.root, self.imlist["img_path"][index])
        with open(file=full_imname, mode="rb") as image_file:
            img = self.turbo_jpeg.decode(image_file.read())
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.with_bbox:
            x, y, w, h = self.imlist.loc[index, "bbox_x":"bbox_h"]
            img = img[y : y + h, x : x + w, :]

        # img = Image.fromarray(img)
        # img = self.transforms(img)
        img = self.transforms(image=img)["image"]
        return img

    def __len__(self):
        return len(self.imlist)


class ValidationWBDataset(data.Dataset):
    def __init__(self, root, annotation_file, transforms, with_bbox=False):
        self.root = root
        self.imlist = pd.read_csv(annotation_file)
        # cvs=read_csv('sample.csv')
        # pd.set_option('display.max_rows', 50)
        # print(self.imlist)
        # print("len annotation_file ", len(self.imlist))

        self.transforms = transforms
        self.with_bbox = with_bbox
        self.turbo_jpeg = TurboJPEG()

    def __getitem__(self, index):
        cv2.setNumThreads(120)

        full_imname = os.path.join(self.root, self.imlist["img_path"][index])
        with open(file=full_imname, mode="rb") as image_file:
            img = self.turbo_jpeg.decode(image_file.read())
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.with_bbox:
            x, y, w, h = self.imlist.loc[index, "bbox_x":"bbox_h"]
            img = img[y : y + h, x : x + w, :]

        # img = Image.fromarray(img)
        # img = self.transforms(img)
        img = self.transforms(image=img)["image"]
        return img

    def __len__(self):
        return len(self.imlist)


class AmazonDataset(data.Dataset):
    def __init__(
        self,
        root,
        annotation_file,
        config
    ):
        self.root = root
        try:
            table = pd.read_csv(annotation_file, delimiter=";")
            # print(table)
            paths = table["img_path"].tolist()
        except:
            table = pd.read_csv(annotation_file, delimiter=",")
            print(table)
            paths = table["img_path"].tolist()
        self.is_galleries = ['gallery' in x for x in paths]
        self.paths = [f"{self.root}{x}" for x in paths]
        labels = table[["class_id", "class_id"]].to_numpy()
        
        self.classes_count = len(table['class_id'].unique())
        print("длина датасета", len(table))
        self.groups_count = len(table['class_id'].unique())
        # print(classes_count, group_count)
        # exit(1)
        type_ids = table[["type_id"]].to_numpy(dtype=np.int32)
        self.labels = set_labels_to_range(labels)
        
        self.labels = np.hstack([self.labels, type_ids])
        # print(self.labels)
        # exit(1)
        # self.labels = self.imlist[["category_id", "class_id"]].to_numpy()

        self.gallery_transform = get_gallery_transform(config)
        self.query_transform = get_query_transform(config)
        self.turbo_jpeg = TurboJPEG()
        # self.feature_extractor = feature_extractor

    def __getitem__(self, index):
        cv2.setNumThreads(120)

        full_imname = self.paths[index]
        category_id, class_id, type_id= self.labels[index]
        is_gallery = self.is_galleries[index]
        # impath, class_id, category_id, type_id = self.imlist.iloc[index]

        # full_imname = os.path.join(self.root, impath)
        # img = read_image(full_imname)
        try:
            with open(file=full_imname, mode="rb") as image_file:
                img = self.turbo_jpeg.decode(image_file.read())
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as ex:
            print(full_imname)
            return torch.zeros(3, 224, 224), category_id, class_id, is_gallery

        transform = self.gallery_transform if type_id == 0 else self.query_transform
        full_imname = Path(full_imname)
        box_txt_path = Path(f"{full_imname.parent}/{full_imname.stem}.txt")
        if type_id == 1 and Path.exists(box_txt_path):

            with box_txt_path.open("r") as file:
                contents = file.readline()
                x1, y1, x2, y2 = contents.split(" ")

                img = img[int(y1):int(y2), int(x1):int(x2), :]
                # with open("images.txt", mode="a") as f:
                #     f.write(str(full_imname) + "/n")
                # cv2.imwrite(f"{full_imname.stem}.jpg", img)
                # exit(1)
        
        img = transform(image=img)["image"]


        return img, category_id, class_id, is_gallery

    def __len__(self):
        return len(self.paths)


if __name__ == '__main__':
    import yaml
    from collections import namedtuple
    
    cfg = '/home/cv_user/visual-product-recognition-2023-giga-flex/MCS2023_baseline/config/clean_wb.yml'
    with open(cfg) as f:
        dictionary = yaml.safe_load(f)
    def convert_dict_to_tuple(dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                dictionary[key] = convert_dict_to_tuple(value)
        return namedtuple("GenericDict", dictionary.keys())(**dictionary)
    config = convert_dict_to_tuple(dictionary)
    ds = WBFlexDataset('/mnt/wb_products_dataset', '/home/cv_user/clean_annotation_m.csv', config)




