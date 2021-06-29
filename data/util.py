import random
from PIL import ImageOps


def get_patch(img_in, img_tar, patch_size, scale, ix=-1, iy=-1):
    ih, iw = img_in.size
    tp = scale * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    tx, ty = scale * ix, scale * iy

    img_in = img_in.crop((iy, ix, iy + ip, ix + ip))  # [:, iy:iy + ip, ix:ix + ip]
    img_tar = img_tar.crop((ty, tx, ty + tp, tx + tp))  # [:, ty:ty + tp, tx:tx + tp]

    return img_in, img_tar


def get_patch_one(img_in, patch_size, ix=-1, iy=-1):
    ih, iw = img_in.size 
    ip = patch_size

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    img_in = img_in.crop((iy, ix, iy + ip, ix + ip))  # [:, iy:iy + ip, ix:ix + ip]

    return img_in


def augment(img_in, img_tar, flip_h=True, rot=True):
    if random.random() < 0.5 and flip_h:
        img_in = ImageOps.flip(img_in)
        img_tar = ImageOps.flip(img_tar)
    
    if random.random() < 0.5 and rot:
        img_in = img_in.rotate(90)
        img_tar = img_tar.rotate(90)
    
    return img_in, img_tar


def modcrop(img, modulo):
    ih, iw = img.size
    ih = ih - ih % modulo
    iw = iw - iw % modulo
    img = img.crop((0, 0, ih, iw))
    return img

