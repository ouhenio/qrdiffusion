import requests
import os
from PIL import Image
from datasets import Dataset, load_dataset
from urllib.parse import urlparse

def download_image(url, save_dir):
    try:
        # get image filename
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        save_path = os.path.join(save_dir, filename)

        # check if image was already downloaded, else download it
        if os.path.exists(save_path):
            print(f"The file {filename} already exists.")
        else:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(save_path, 'wb') as file:
                    file.write(response.content)
            else:
                print(f"Couldn't download image from URL: {url}")
                return None
        return save_path

    except Exception as e:
        print(f"Error downloading image from URL: {url}")
        print(f"Error message: {str(e)}")
        return None
    

class ImprovedAestheticsDataloader:
    def __init__(self):
        pass

    def load_hf_dataset(self, split="train"):
        self.hf_dataset = load_dataset(
            "ChristophSchuhmann/improved_aesthetics_6.5plus",
            split=split,
        )

    def prepare_images(self):
        assert self.hf_dataset, "There's no dataset to get images from."

        # download img and create new dataset
        self.hf_dataset.map()