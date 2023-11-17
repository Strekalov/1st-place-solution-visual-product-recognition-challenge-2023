import pandas as pd

import torch
from tqdm import tqdm

from pathlib import Path
import typer

from loguru import logger

# from sklearn.metrics.cluster import normalized_mutual_info_score
# from cuml.neighbors import KNeighborsClassifier
device = "cuda"
# /mnt/wb_products_dataset


def clear_class(embeddings, labels, img_path_list, df_full):
    embeddings.to(device)
    output = torch.cdist(embeddings, embeddings)
    output.fill_diagonal_(4)
    npp = torch.argwhere(output < 0.2)
    # stnpp = set()
    # for (x1, x2) in npp.numpy():
    #     if (x1, x2) not in stnpp and (x2, x1) not in stnpp:
    #         stnpp.add((x1, x2))
    delete_set = set()
    save_set = set()
    # print(stnpp)
    for elem in npp:
        file_name1 = str(
            img_path_list[elem[0]]
        )  # .replace('/home/cv_user/projects/mcs23s/data/mcs23', '')
        file_name2 = str(
            img_path_list[elem[1]]
        )  # .replace('/home/cv_user/projects/mcs23s/data/mcs23', '')
        if str(file_name1) == str(file_name2):
            # print('same',file_name1)
            continue
        elif str(file_name1) in save_set:
            delete_set.add(str(file_name2))
            # print('file_name2', file_name2)
        elif str(file_name2) in save_set:
            delete_set.add(str(file_name1))
            # print('file_name1', file_name1)
        else:
            save_set.add(str(file_name1))
            delete_set.add(str(file_name2))
            print(img_path_list[elem[0]], img_path_list[elem[1]])
        # print('+'*80)
    # print(len(df_full))
    for elem in delete_set:
        df_full = df_full.drop(df_full[df_full["img_path"] == elem].index)

    return df_full


def filter_annotation(csv_path, dataset_path, output_path):
    dataset_root = Path(dataset_path)
    df_full = pd.read_csv(csv_path, sep=";")
    category_ids = list(set(df_full["category_id"]))
    # print(category_ids)
    for category_id in category_ids:
        df = df_full[df_full["category_id"] == category_id]
        # print('len df', len(df))

        df = df[df["img_path"].apply(lambda x: "gallery" in x)]
        # len_df = len(df)
        # vc = df['class_id'].value_counts()

        # print('df len', len_df)

        temp = list(set(df["class_id"]))
        class2id = {x: i for i, x in enumerate(temp)}
        # id2class = {i:x for i, x in enumerate(temp)}
        # print('len classes', len(class2id))

        for class_name in class2id.keys():
            labels = []
            img_path_list = []
            temp_df = df[df.class_id == class_name]
            embeddings = torch.empty((len(temp_df), 2048))
            for i in range(len(temp_df)):
                embedding_path = dataset_root / temp_df.iloc[i]["img_path"][1:].replace(
                    ".jpg", ".pt"
                )
                embeddings[i] = torch.load(embedding_path, map_location=device)
                labels.append(class2id[class_name])
                img_path_list.append(temp_df.iloc[i]["img_path"])
                # print(labels)
                # print(embeddings.shape)
                # print(img_path_list)
            df_full = clear_class(embeddings, labels, img_path_list, df_full)

            # exit(0)
    df_full.to_csv(output_path, index=None)


def main(
    csv_path: Path = typer.Option(help="Path to csv annotation"),
    dataset_path: Path = typer.Option(help="Path to dataset"),
    output_path: Path = typer.Option(help="Path to output csv after filtering"),
):
    try:
        filter_annotation(csv_path, dataset_path, output_path)
    except Exception as ex:
        logger.exception(ex)


if __name__ == "__main__":
    try:
        typer.run(main)
    except SystemExit:
        typer.secho("Программа завершена", fg=typer.colors.BRIGHT_GREEN)
