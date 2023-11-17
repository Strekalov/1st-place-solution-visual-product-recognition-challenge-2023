from pathlib import Path
from typing import Tuple, List
import pandas as pd
import typer
from tqdm import tqdm
import torch
import cv2
from albumentations.pytorch import ToTensorV2
import albumentations as A
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from loguru import logger


class EmbeddingExtractor:
    def __init__(self, model_path, device) -> None:

        self.device = device
        self.model = torch.jit.load(model_path)
        self.model.half()
        self.model.eval()
        self.model.to(self.device)

    def extract_embedding(self, images):
        with torch.inference_mode():
            return self.model(images)


class ImageProcessor:
    @staticmethod
    def np_resize(image, width=None, height=None, inter=cv2.INTER_AREA):

        dim = None
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        elif height is None:
            r = width / float(w)
            dim = (width, int(h * r))
        else:
            dim = (width, height)
        resized = cv2.resize(image, dim, interpolation=inter)
        return resized

    @staticmethod
    def _get_transform():
        return A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    def image_to_tensor(self, image):

        transform = self._get_transform()
        img = transform(image=image)["image"]
        img = img.unsqueeze(0)
        return img


class ProductDetector:
    def __init__(
        self,
        image_processor: ImageProcessor,
        embedding_extractor: EmbeddingExtractor,
        width: int = 600,
        pyramid_scale: float = 1.6,
        window_step: int = 32,
        roi_size: Tuple[int, int] = (256, 256),
        input_size: Tuple[int, int] = (256, 256),
    ) -> None:
        self.image_processor = image_processor
        self.embeddig_extractor = embedding_extractor
        self.device = self.embeddig_extractor.device
        self.width = width
        self.pyramid_scale = pyramid_scale
        self.window_step = window_step
        self.roi_size = roi_size
        self.input_size = input_size

    def _sliding_window(self, image):

        for y in range(0, image.shape[2] - self.roi_size[1], self.window_step):
            for x in range(0, image.shape[3] - self.roi_size[0], self.window_step):
                yield (
                    x,
                    y,
                    image[:, :, y : y + self.roi_size[1], x : x + self.roi_size[0]],
                )

    def _image_pyramid(self, image):

        yield image

        while True:
            w = int(image.shape[1] / self.pyramid_scale)
            image = self.image_processor.np_resize(image, width=w)

            if image.shape[0] < self.roi_size[1] or image.shape[1] < self.roi_size[0]:
                break

            yield image

    def _calc_windows_count(self, image: np.ndarray):
        images_count = 0
        pyramid = self._image_pyramid(image)
        for scaled_image in pyramid:
            scaled_image = self.image_processor.image_to_tensor(scaled_image)
            for _ in self._sliding_window(scaled_image):
                images_count += 1
        return images_count

    def find_most_simular(self, query_image, gallery_embeddings):
        query_image = cv2.resize(
            query_image, self.input_size, interpolation=cv2.INTER_AREA
        )

        query_image_tensor = (
            self.image_processor.image_to_tensor(query_image).half().to(self.device)
        )

        query_image_embedding = self.embeddig_extractor.extract_embedding(
            query_image_tensor
        )

        dist = F.cosine_similarity(query_image_embedding, gallery_embeddings.float())
        best_simularity_idx = torch.argmax(dist)

        return gallery_embeddings[best_simularity_idx]

    def _detect_box(self, query_image: np.ndarray, gallery_embedding: torch.Tensor):

        resized_query_image = self.image_processor.np_resize(
            query_image, width=self.width
        )
        (H, W) = resized_query_image.shape[:2]

        W_SCALE = query_image.shape[1] / float(W)
        H_SCALE = query_image.shape[0] / float(H)
        rois = []
        locs = []
        sims = []

        windows_count = self._calc_windows_count(resized_query_image)
        if windows_count == 0:
            raise ValueError("windows_count is 0!")
        batch_images = torch.zeros(
            (windows_count, 3, 256, 256),
            device=self.device,
            requires_grad=False,
            dtype=torch.half,
        )

        pyramid = self._image_pyramid(resized_query_image)
        i = 0
        for image in pyramid:

            scale = W / float(image.shape[1])
            image = self.image_processor.image_to_tensor(image).half().to(self.device)

            for (x, y, roi) in self._sliding_window(image):

                x = int(x * scale)
                y = int(y * scale)
                w = int(self.roi_size[0] * scale)
                h = int(self.roi_size[1] * scale)

                batch_images[i] = roi
                i += 1
                locs.append((x, y, x + w, y + h))

        roi_embedding = self.embeddig_extractor.extract_embedding(batch_images)
        sims = F.cosine_similarity(gallery_embedding.float(), roi_embedding.float())

        sims = sims.cpu().numpy()
        boxes = np.asarray(locs)
        scores = np.asarray(sims)
        max_simularity = np.max(scores)

        boxes_args = np.argwhere(scores >= max_simularity * 0.95)
        boxes_args = boxes_args.reshape(-1)
        final_boxes = boxes[boxes_args]

        x1 = int(min(final_boxes[:, 0]) * W_SCALE)
        y1 = int(min(final_boxes[:, 1]) * H_SCALE)
        x2 = int(max(final_boxes[:, 2]) * W_SCALE)
        y2 = int(max(final_boxes[:, 3]) * H_SCALE)

        return x1, y1, x2, y2

    def detect_box(self, query_image: np.ndarray, gallery_images: torch.Tensor):

        most_simular_gallery_embedding = self.find_most_simular(
            query_image, gallery_images
        )

        return self._detect_box(query_image, most_simular_gallery_embedding)


def get_tensor_image_batch(image_paths: List[str], size=None):

    images_array = None
    filtered_image_paths = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if image is None:
            continue

        if size is not None:
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        img_proc = ImageProcessor()
        image = img_proc.image_to_tensor(image)
        if images_array is None:
            images_array = image
        else:
            images_array = torch.vstack((images_array, image))
        filtered_image_paths.append(image_path)

    return images_array, filtered_image_paths


def get_query_images(image_paths: List[str], size=None):

    images_list = []
    filtered_image_paths = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if image is None:
            continue

        if size is not None:
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)

        images_list.append(image)

        filtered_image_paths.append(image_path)

    return images_list, filtered_image_paths


def write_box(image_path: Path, x1, y1, x2, y2):
    image_path = Path(image_path)
    txt_file_path = f"{image_path.parent}/{image_path.stem}.txt"
    with open(txt_file_path, "w") as file:
        file.write(f"{x1} {y1} {x2} {y2}")


def find_boxes(model_path, dataset_path, csv_path, gpu=1):

    image_processor = ImageProcessor()
    device = torch.device(f"cuda:{gpu}")
    embedding_extractor = EmbeddingExtractor(
        model_path=model_path, device=device
    )

    BOX_DETECTOR = ProductDetector(
        image_processor=image_processor, embedding_extractor=embedding_extractor
    )

    dataset_path = Path(dataset_path)
    df = pd.read_csv(csv_path, delimiter=";", index_col=False)

    unique_products = df.class_id.unique()

    for product in tqdm(unique_products):
        product_images_df = df[df["class_id"] == product]
        query_images_df = product_images_df[product_images_df["type_id"] == 1]
        gallery_images_df = product_images_df[product_images_df["type_id"] == 0]

        query_image_paths = query_images_df["img_path"].tolist()
        gallery_image_paths = gallery_images_df["img_path"].tolist()

        query_image_paths = [
            Path(f"{dataset_path}{query_path}") for query_path in query_image_paths
        ]
        gallery_image_paths = [
            Path(f"{dataset_path}{gallery_path}")
            for gallery_path in gallery_image_paths
        ]
        if len(query_image_paths) == 0:
            logger.info(f"У продукта {product} нет фото отзывов.")
            continue
        if len(gallery_image_paths) == 0:
            logger.info(f"У продукта {product} нет фото из галлереи.")
            continue
        try:
            query_images_list, query_paths = get_query_images(query_image_paths)

            gallery_images_tensor, gallery_paths = get_tensor_image_batch(
                gallery_image_paths, size=BOX_DETECTOR.input_size
            )

            gallery_images_tensor = gallery_images_tensor.half().to(BOX_DETECTOR.device)
            gallery_embeddings = BOX_DETECTOR.embeddig_extractor.extract_embedding(
                gallery_images_tensor
            )

            for query_image, query_path in zip(query_images_list, query_paths):
                if not Path.exists(Path(f"{query_path.parent}/{query_path.stem}.txt")):
                    try:
                        x1, y1, x2, y2 = BOX_DETECTOR.detect_box(
                            query_image, gallery_embeddings
                        )
                    except Exception as ex:
                        logger.error(
                            f"Ошибка в поиске бокса: {ex}\nИзображение: {str(query_path)}"
                        )
                        continue
                    write_box(query_path, x1, y1, x2, y2)
        except Exception as ex:
            logger.exception(ex)
            continue

    logger.info(f"Боксы посчитались! {csv_path}")


def main(
    model_path: Path = typer.Option(help="Path to model"),
    dataset_path: Path = typer.Option(help="path to dataset"),
    csv_path: Path = typer.Option(help="path to csv"),
    gpu: int = typer.Option(default=0, help="GPU device for use"),
):
    try:
        find_boxes(model_path, dataset_path, csv_path, gpu)
    except Exception as ex:
        logger.exception(ex)


if __name__ == "__main__":
    try:
        typer.run(main)
    except SystemExit:
        typer.secho("Программа завершена", fg=typer.colors.BRIGHT_GREEN)
