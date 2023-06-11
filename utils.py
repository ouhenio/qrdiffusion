from typing import Union

import qrcode
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
    alpha: float,
) -> Image.Image:
    qr_image = create_qr(url)
    qr_image = qr_image.resize(image.size, Image.ANTIALIAS)

    # convert images to rgba to support alpha channel
    qr_image = qr_image.convert("RGBA")
    image = image.convert("RGBA")

    # apply overlay
    overlayed_image = Image.blend(image, qr_image, alpha=alpha)
    
    return overlayed_image
