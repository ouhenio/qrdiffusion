import requests
import os
from PIL import Image
from datasets import Dataset, load_dataset
from urllib.parse import urlparse

def download_image(url, save_dir):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            # get image filename
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)
            save_path = os.path.join(save_dir, filename)
            
            with open(save_path, 'wb') as file:
                file.write(response.content)
            
            return save_path
    except Exception as e:
        print(f"Error downloading image from URL: {url}")
        print(f"Error message: {str(e)}")
        return None