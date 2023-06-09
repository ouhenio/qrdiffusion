import os
import random
import string
from typing import Optional, Union
from urllib.parse import urlparse

import numpy as np
import qrcode
import qrcode.image.svg
import requests
from PIL import Image


def create_qr(url: str) -> Image.Image:
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)

    qr_image = qr.make_image(fill="black", back_color="white")

    return qr_image


def overlay_qr(
    url: str,
    image: Union[Image.Image, str],
    background_alpha: int = 60,
    qr_alpha: int = 200,
) -> Image.Image:
    image = image.convert("RGBA")

    qr_image = create_qr(url)
    qr_image = qr_image.resize(image.size, Image.ANTIALIAS).convert("RGBA")

    # convert to numpy to parallelize alpha processing
    np_qr_image = np.array(qr_image)
    is_white = np.all(np_qr_image[:, :, :3] > 200, axis=2)
    np_qr_image[is_white, 3] = background_alpha  # apply background_alpha to white
    np_qr_image[~is_white, 3] = qr_alpha  # apply qr_alpha to qr content

    qr_transparent = Image.fromarray(np_qr_image)

    image.paste(qr_transparent, (0, 0), qr_transparent)

    return image


def download_image_from_url(
    url: str,
    save_dir: str,
) -> Optional[str]:
    try:
        # get image filename and save path
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        save_path = os.path.join(save_dir, filename)

        # check if image was already downloaded, otherwise download it
        if os.path.exists(save_path):
            print(f"The file {save_path} already exists.")
        else:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(save_path, "wb") as file:
                    file.write(response.content)
            else:
                print(f"Couldn't download image from URL: {url}")
                return None

        # crop image to its smallest dimension
        image = Image.open(save_path)
        width, height = image.size
        min_dim = min(width, height)
        left = (width - min_dim) / 2
        top = (height - min_dim) / 2
        right = (width + min_dim) / 2
        bottom = (height + min_dim) / 2

        image = image.crop((left, top, right, bottom))

        # save cropped image
        image.save(save_path)

        return save_path

    except Exception as e:
        print(f"Error downloading image from URL: {url}")
        print(f"Error message: {str(e)}")
        return None


def generate_random_string(length: int = 12) -> str:
    return "".join(
        random.choice(string.ascii_letters + string.digits) for _ in range(length)
    )
