import os
import random
from pathlib import Path

import datasets
from datasets import load_dataset
from dotenv import load_dotenv
from PIL import Image
from torchvision import transforms

from utils import create_qr, download_image_from_url, generate_random_string, overlay_qr

load_dotenv()

image_transforms = transforms.Compose(
    [
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(512),
    ]
)

target_path = os.environ.get("TARGET_PATH")
map_cache_target_path = os.environ.get("MAP_CACHE_PATH")
datasets.config.DOWNLOADED_DATASETS_PATH = Path(target_path)


class DiffussionDB:
    caption_key = "prompt"
    image_key = "image"
    qr_key = "qr"

    def __init__(
        self,
        version="2m_random_10k",
        split="train",
    ):
        self.dataset = load_dataset(
            "poloclub/diffusiondb",
            version,
            cache_dir=target_path,
            split=split,
        )

    def create_qr_image(self, element):
        string_length = random.randint(6, 24)
        url = generate_random_string(string_length)
        image = element[self.image_key]
        image = image_transforms(image)
        qr_image = overlay_qr(url=url, image=image)

        qr = create_qr(url)
        qr = qr.resize(image.size, Image.ANTIALIAS)

        return {self.image_key: qr_image, self.qr_key: qr}

    def prepare_data(self):
        self.dataset = self.dataset.map(
            self.create_qr_image,
            cache_file_name=map_cache_target_path,
        )


class ImprovedAestheticsDataloader:
    image_url_key = "URL"
    caption_key = "TEXT"
    image_path_key = "image_path"
    qr_image_path_key = "qr_path"

    def __init__(
        self,
        split: str = "train",
        images_folder: str = "images",
        qr_images_folder: str = "qr_images",
    ) -> None:
        self.images_folder = images_folder
        if not os.path.exists(images_folder):
            os.mkdir(images_folder)
        self.qr_images_folder = qr_images_folder
        if not os.path.exists(qr_images_folder):
            os.mkdir(qr_images_folder)
        self.load_hf_dataset(split)

    def load_hf_dataset(self, split: str) -> None:
        self.dataset = load_dataset(
            "ChristophSchuhmann/improved_aesthetics_6.5plus",
            split=split,
        )

    def download_image(self, element):
        image_path = download_image_from_url(
            element[self.image_url_key], self.images_folder
        )
        return image_path

    def create_qr_images(self, element):
        # create random string
        url = generate_random_string()

        # load image and create qr path
        image_path = element[self.image_path_key]
        image_filename = os.path.basename(image_path)
        qr_image_path = os.path.join(self.qr_images_folder, image_filename)

        # make overlay and save it
        image = Image.open(image_path)
        qr_img = overlay_qr(url=url, image=image)
        qr_img.save(qr_image_path, format="JPEG")

        # create qr column
        return qr_image_path

    def prepare_data(self):
        assert self.dataset, "There's no dataset to get images from."

        dataset = self.dataset.map(
            lambda element: {self.image_path_key: self.download_image(element)}
        )
        dataset = dataset.filter(
            lambda element: element[self.image_path_key] is not None
        )
        dataset = dataset.map(
            lambda element: {self.qr_image_path_key: self.create_qr_images(element)}
        )

        self.dataset = dataset
