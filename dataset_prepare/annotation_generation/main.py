import csv
from pathlib import Path
from typing import Union

import typer
from tqdm import tqdm
import torch
import torch.nn.functional as F


class CSVAnnotations:
    def __init__(self, file_path: str = "annotations.csv"):
        self.fieldnames = ["img_path", "class_id", "category_id", "type_id"]
        self.file_path = Path(file_path)
        self.file_path.touch(exist_ok=True)
        with open(self.file_path, "w") as csv_file:
            self.writer = csv.DictWriter(
                csv_file, fieldnames=self.fieldnames, delimiter=";"
            )
            self.writer.writeheader()

    def write(
        self,
        image_path: str,
        class_id: Union[int, str],
        category_id: Union[int, str],
        type_id: int,
    ):
        with open(self.file_path, "a") as csv_file:
            self.writer = csv.DictWriter(
                csv_file, fieldnames=self.fieldnames, delimiter=";"
            )
            self.writer.writerow(
                {
                    "img_path": image_path,
                    "class_id": class_id,
                    "category_id": category_id,
                    "type_id": type_id,
                }
            )


def filter_images(image_dir: Path):
    tensors_files = list(image_dir.rglob("*.pt"))
    tensors_list = []
    file_paths = []
    filtered_images = []
    tensors = []
    for tensor_file in tensors_files:
        try:
            tensor_b = torch.load(str(tensor_file), map_location="cpu")

        except Exception as ex:
            print(f"Failed load tensor: {tensor_file}")
            continue
        tensor_b = tensor_b.unsqueeze(0)
        tensors_list.append(tensor_b)
        file_paths.append(tensor_file.__str__())
        tensors = torch.cat(tensors_list, dim=0).float()
    for i in range(len(tensors)):

        dist = F.cosine_similarity(tensors[i], tensors.float())
        dist = dist[dist <= 0.99]
        median = torch.mean(dist)

        if median < 0.3:
            continue

        filtered_images.append(str(file_paths[i])[:-2] + "jpg")
    return filtered_images


def generate_annotations(input_path: str, output_path: str):
    # output_dir = Path(output_path)
    # output_dir.mkdir(exist_ok=True)

    annotator = CSVAnnotations(file_path=output_path)
    classes_count = 0
    category_count = 0
    for type_dir in tqdm(list(Path(input_path).iterdir())):
        if not type_dir.is_dir():
            continue
        category_dirs = list(type_dir.iterdir())
        for category_dir in tqdm(category_dirs, desc="Categories..."):
            product_dirs = list(category_dir.iterdir())
            for product_dir in tqdm(product_dirs, desc="Classes..."):

                # image_paths = list(product_dir.rglob("*.jpg"))

                image_paths = filter_images(product_dir)

                if len(image_paths) == 0:
                    continue
                # if len(image_paths) < 3:
                #     continue
                type_id = 0 if type_dir.name == "gallery" else 1
                for image_path in image_paths:
                    relation_path = str(image_path).split(str(input_path))[1]
                    annotator.write(
                        image_path=relation_path,
                        category_id=category_dir.name,
                        class_id=product_dir.name,
                        type_id=type_id,
                    )
                classes_count += 1
            category_count += 1
    typer.secho(f"Всего классов: {classes_count}", fg=typer.colors.BRIGHT_YELLOW)
    typer.secho(f"Всего категорий: {classes_count}", fg=typer.colors.BRIGHT_YELLOW)
    typer.secho(f"Сохранено в {output_path}")


def main(
    dataset_path: Path = typer.Argument(
        "/mnt/wb_products_dataset/", help="Путь до папки с датасетом"
    ),
    annotation_path: Path = typer.Argument(
        "annotations_wb_artems_script.csv",
        help="Путь до выходного файла",
    ),
):
    generate_annotations(dataset_path, annotation_path)


if __name__ == "__main__":
    try:
        typer.run(main)
    except SystemExit:
        typer.secho("Программа завершена", fg=typer.colors.BRIGHT_GREEN)
