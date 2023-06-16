import requests
import os
import random
import string
import io
import cairosvg


from typing import Union, Optional

import qrcode
import qrcode.image.svg
from PIL import Image
from urllib.parse import urlparse


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

def create_svg_qr(url: str) -> Image.Image:
    factory = qrcode.image.svg.SvgImage
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)

    qr_image = qr.make_image(
        image_factory=factory,
        fill="black",
        back_color="none",
    )

    png_io = io.BytesIO()
    cairosvg.svg2png(
        bytestring=qr_image.to_string(),
        write_to=png_io,
        output_width=512,
        output_height=512,
    )

    qr_image = Image.open(io.BytesIO(png_io.getvalue()))

    return qr_image

def overlay_qr(
    url: str,
    image: Union[Image.Image, str],
    alpha: float,
) -> Image.Image:
    qr_image = create_qr(url)
    qr_image = qr_image.resize(image.size, Image.ANTIALIAS).convert("RGBA")

    svg_qr_image = create_svg_qr(url)
    svg_qr_image = svg_qr_image.resize(image.size, Image.ANTIALIAS).convert("RGBA")

    # apply overlay
    overlayed_image = Image.blend(image, qr_image, alpha=alpha)
    overlayed_image.paste(svg_qr_image, (0, 0), svg_qr_image)
    
    return overlayed_image

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
                with open(save_path, 'wb') as file:
                    file.write(response.content)
            else:
                print(f"Couldn't download image from URL: {url}")
                return None
            
        # crop image to its smallest dimension
        image = Image.open(save_path)
        width, height = image.size
        min_dim = min(width, height)
        left = (width - min_dim)/2
        top = (height - min_dim)/2
        right = (width + min_dim)/2
        bottom = (height + min_dim)/2

        image = image.crop((left, top, right, bottom))

        # save cropped image
        image.save(save_path)

        return save_path

    except Exception as e:
        print(f"Error downloading image from URL: {url}")
        print(f"Error message: {str(e)}")
        return None
    
def generate_random_string(length: int = 12) -> str:
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))