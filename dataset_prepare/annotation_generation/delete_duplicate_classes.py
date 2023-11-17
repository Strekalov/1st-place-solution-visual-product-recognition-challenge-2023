import pandas as pd

import typer
import torch

import torch

from loguru import logger
import time

from pathlib import Path


def filter_annotation(csv_path, dataset_path, output_path):
    # from sklearn.metrics.cluster import normalized_mutual_info_score
    # from cuml.neighbors import KNeighborsClassifier
    device = "cpu"
    # /mnt/wb_products_dataset
    dataset_root = Path(dataset_path)
    df_full = pd.read_csv(
        csv_path, sep=";"
    )
    print("len df", len(df_full))
    category_ids = list(set(df_full["category_id"]))
    print(category_ids)
    for category_id in category_ids:
        df = df_full[df_full["category_id"] == category_id]
        # print('len df', len(df))

        df = df[df["img_path"].apply(lambda x: "gallery" in x)]
        len_df = len(df)
        vc = df["class_id"].value_counts()

        # print('df len', len_df)
        embeddings = torch.empty((len_df, 2048))
        temp = list(set(df["class_id"]))
        class2id = {x: i for i, x in enumerate(temp)}
        # id2class = {i:x for i, x in enumerate(temp)}
        # print('len classes', len(class2id))
        j = 0
        labels = []
        img_path_list = []
        for class_name in class2id.keys():
            temp_df = df[df.class_id == class_name]
            for i in range(len(temp_df)):
                embedding_path = dataset_root / temp_df.iloc[i]["img_path"][1:].replace(
                    ".jpg", ".pt"
                )
                embeddings[j] = torch.load(embedding_path, map_location=device)
                labels.append(class2id[class_name])
                img_path_list.append(dataset_root / temp_df.iloc[i]["img_path"][1:])
                j += 1
        # print(labels)
        embeddings.to(device)
        # cos = nn.CosineSimilarity(eps=1e-6)
        strt = time.time()
        output = torch.cdist(embeddings, embeddings)
        print(time.time() - strt)
        # print(output)
        # np_out = output.numpy()
        # sns_plot = sns.heatmap(np_out< 0.2)
        output.fill_diagonal_(4)
        # print(np.argwhere(np_out< 0.2 ))
        npp = torch.argwhere(output < 0.2)
        stnpp = set()
        for (x1, x2) in npp.numpy():
            if (x1, x2) not in stnpp and (x2, x1) not in stnpp:
                stnpp.add((x1, x2))
        # print(stnpp)
        # sns_plot.figure.savefig("output.png")
        delete_set = set()
        save_set = set()
        pairs = []
        for elem in stnpp:

            subfolder1 = img_path_list[elem[0]].parent
            subfolder2 = img_path_list[elem[1]].parent
            if str(subfolder1) == str(subfolder2):
                continue
            elif str(subfolder1) in save_set:
                delete_set.add(str(subfolder2))
            elif str(subfolder2) in save_set:
                delete_set.add(str(subfolder1))
            else:
                save_set.add(str(subfolder1))
                delete_set.add(str(subfolder2))
                print(subfolder1, subfolder2)
            # print('+'*80)
        print()
        print(delete_set)
        # print(save_set)
        for elem in delete_set:
            class_id = elem.split("/")[-1]
            df_full = df_full.drop(df_full[df_full["class_id"] == class_id].index)
        # df = df_full[df_full['category_id'] == 128.0]
        print("<->" * 50)
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
