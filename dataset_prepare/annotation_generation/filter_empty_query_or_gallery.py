import pandas as pd
from tqdm import tqdm
import typer
from pathlib import Path
from loguru import logger



def filter_annotation(csv_path, output_path):


    df = pd.read_csv(csv_path, delimiter=";", index_col=False)

    unique_products = df.class_id.unique()

    dfs_list = []
    bad_class_count = 0
    for product in tqdm(unique_products):
            product_images_df = df[df["class_id"]==product]
            query_images_df = product_images_df[product_images_df["type_id"]==1]
            gallery_images_df = product_images_df[product_images_df["type_id"]==0]
            
            query_image_paths = query_images_df["img_path"].tolist()
            gallery_image_paths = gallery_images_df["img_path"].tolist()
            

            if len(query_image_paths) == 0 or len(gallery_image_paths) == 0:
                bad_class_count+=1
                continue
            
            dfs_list.append(product_images_df)
            

    final_df = pd.concat(dfs_list)

    print(bad_class_count)
    print(len(final_df))
    print(len(df))
    final_df.to_csv(output_path, sep=";", index=False)




def main(
    csv_path: Path = typer.Option(help="Path to csv annotation"),
    output_path: Path = typer.Option(help="Path to output csv after filtering"),
):
    try:
        filter_annotation(csv_path, output_path)
    except Exception as ex:
        logger.exception(ex)


if __name__ == "__main__":
    try:
        typer.run(main)
    except SystemExit:
        typer.secho("Программа завершена", fg=typer.colors.BRIGHT_GREEN)