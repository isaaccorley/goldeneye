from urllib.request import urlopen

import numpy as np
import torchvision.transforms as transforms
from PIL import Image, ImageFont


def get_font():
    truetype_url = "https://huggingface.co/internlm/internlm-xcomposer2d5-7b/resolve/main/SimHei.ttf?download=true"
    ff = urlopen(truetype_url)
    font = ImageFont.truetype(ff, size=40)
    return font


def padding_336(b, pad=336):
    width, height = b.size
    tar = int(np.ceil(height / pad) * pad)
    top_padding = 0  # int((tar - height)/2)
    bottom_padding = tar - height - top_padding
    left_padding = 0
    right_padding = 0
    b = transforms.functional.pad(
        b, [left_padding, top_padding, right_padding, bottom_padding], fill=[255, 255, 255]
    )

    return b


def Image_transform(img, hd_num=25):
    print(f"HD is {hd_num}")
    width, height = img.size
    print(f"original HW : {width} {height}")
    trans = False
    if width < height:
        img = img.transpose(Image.TRANSPOSE)
        trans = True
        width, height = img.size
    ratio = width / height
    scale = 1
    while scale * np.ceil(scale / ratio) <= hd_num:
        scale += 1
    scale -= 1
    scale = max(np.floor(width / 560), scale)
    new_w = int(scale * 560)
    new_h = int(new_w / ratio)

    img = transforms.functional.resize(
        img,
        [new_h, new_w],
    )
    img = padding_336(img, 560)
    width, height = img.size
    print(f"new HW : {width} {height}")
    print(f"number of patches : {int(width / 560 * height / 560)}")
    if trans:
        img = img.transpose(Image.TRANSPOSE)

    return img
