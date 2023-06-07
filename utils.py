import io
from typing import Union

import cairosvg
import qrcode
from PIL import Image
from qrcode.image.svg import SvgImage


def overlay_qr(url: str, image: Union[Image.Image, str]) -> Image.Image:
    if isinstance(image, str):
        image = Image.open(image)

    image = image.convert("RGBA")

    # qr will cover 90% of the image
    qr_size = tuple(int(0.9 * dim) for dim in image.size)

    # create qr
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)

    # parse qr to svg
    svg_qr = qr.make_image(image_factory=SvgImage, fill="black", back_color="none")

    # parse svg to png
    png_io = io.BytesIO()
    cairosvg.svg2png(
        bytestring=svg_qr.to_string(),
        write_to=png_io,
        output_width=qr_size[0],
        output_height=qr_size[1],
    )

    # parse png into PIL
    qr_img = Image.open(io.BytesIO(png_io.getvalue()))

    # convert the qr image to rgba
    if qr_img.mode != "RGBA":
        qr_img = qr_img.convert("RGBA")

    # get coordinates to center qr
    position = (
        (image.width - qr_img.width) // 2,
        (image.height - qr_img.height) // 2,
    )

    # paste qr into image
    image.paste(qr_img, position, qr_img)

    return image
