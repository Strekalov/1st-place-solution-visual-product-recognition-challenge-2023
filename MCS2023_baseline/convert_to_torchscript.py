from pathlib import Path
from loguru import logger
import torch
import typer


from models.model import RetrivealNet, Trunk


def convert_to_torchscript(checkpoint_path, output_path):

    trunk = Trunk(
        backbone="convnext_xxlarge.clip_laion2b_rewind",
        pretrained=False,
        embedding_dim=2048,
    )

    model = RetrivealNet(trunk=trunk)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )["state_dict"]

    model.eval()

    model.load_state_dict(checkpoint)
    model.half()

    with torch.jit.optimized_execution(True):
        model = torch.jit.script(model)
    model.save(output_path)


def main(
    checkpoint_path: Path = typer.Option(help="Path to checkpoint"),
    output_path: Path = typer.Option(help="Path to output file"),
):
    try:
        convert_to_torchscript(checkpoint_path, output_path)
    except Exception as ex:
        logger.exception(ex)


if __name__ == "__main__":
    try:
        typer.run(main)
    except SystemExit:
        typer.secho("Программа завершена", fg=typer.colors.BRIGHT_GREEN)
