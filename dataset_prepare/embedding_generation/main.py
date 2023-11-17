import os

import torch
import typer
from loguru import logger
from tqdm import tqdm

from dataset import ThreshDataset
from pathlib import Path


BATCH_SIZE = 128


def compute_embeddings(model_path, dataset_path):

    model = torch.jit.load(model_path)
    model.half()
    model.eval()
    model.to("cuda")

    dataset = ThreshDataset(root=dataset_path)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, pin_memory=True, num_workers=70
    )

    data_iterator = dataloader

    data_iterator = tqdm(
        data_iterator, desc="Compute embeddings...", dynamic_ncols=True, position=1
    )

    with torch.no_grad():
        for (images, paths) in data_iterator:

            try:
                embeddings = model(images.half().to("cuda"))
            except Exception as ex:
                continue

            for embedding, path in zip(embeddings, paths):
                if path == 0:
                    continue
                path = Path(path)
                new_path = Path(f"{path.parent}/{path.stem}.pt").__str__()
                embedding = embedding.detach().cpu()
                torch.save(embedding, new_path)
                os.chmod(new_path, 0o777)


def main(
    model_path: Path = typer.Option(help="Path to model"),
    dataset_path: Path = typer.Option(help="Path to dataset"),
):
    try:
        compute_embeddings(model_path, dataset_path)
    except Exception as ex:
        logger.exception(ex)


if __name__ == "__main__":
    try:
        typer.run(main)
    except SystemExit:
        typer.secho("Программа завершена", fg=typer.colors.BRIGHT_GREEN)
