import os
from PIL import Image
from datasets import load_dataset
from utils import (
    overlay_qr,
    download_image_from_url,
    generate_random_string,
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
        qr_images_folder: str = 'qr_images',
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
        image_path = download_image_from_url(element[self.image_url_key], self.images_folder)
        return image_path

    def create_qr_images(self, element):
        # create random string
        url = generate_random_string()

        # load image and create qr path
        image_path = element[self.image_path_key]
        print(image_path)
        image_filename = os.path.basename(image_path)
        qr_image_path = os.path.join(self.qr_images_folder, image_filename)

        # make overlay and save it
        image = Image.open(image_path)
        qr_img = overlay_qr(url=url, image=image, alpha=0.3)
        qr_img.save(qr_image_path, format="JPEG")

        # create qr column
        return qr_image_path

    def prepare_data(self):
        assert self.dataset, "There's no dataset to get images from."
    
        print("Downloading images...")
        dataset = self.dataset.map(lambda element: {self.image_path_key: self.download_image(element)})
        dataset = dataset.filter(lambda element: element[self.image_path_key] != None)
        print("Creating QRs...")
        dataset = dataset.map(lambda element: {self.qr_image_path_key: self.create_qr_images(element)})

        self.dataset = dataset