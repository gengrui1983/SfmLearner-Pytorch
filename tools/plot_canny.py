import argparse
import os

from path import Path

import cv2

import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("dataset_dir", metavar='DIR',
                    help='path to original dataset')

parser.add_argument("--dump-boundary-root", type=str, default='dump', help="Where to dump the data")
parser.add_argument("--height", type=int, default=128, help="image height")
parser.add_argument("--width", type=int, default=416, help="image width")

args = parser.parse_args()


def boundary(raw_input):
    """
    calculate boundary mask & save
    :param raw_input: *instanceIds image
    :param save_path: city name
    :param save_name: boundary mask name
    :return:
    """
    # process instance mask
    instance_mask = Image.open(raw_input)
    width = instance_mask.size[0]
    height = instance_mask.size[1]

    mask_array = np.array(instance_mask)

    # define the boundary mask
    boundary_mask = np.zeros((height, width), dtype=np.uint8)  # 0-255

    # perform boundary calculate: the center pixel_id is differ from the 4-nearest pixels_id
    for i in range(1, height-1):
        for j in range(1, width-1):
                if not color_same(mask_array[i, j], mask_array[i - 1, j]) \
                        or not color_same(mask_array[i, j], mask_array[i + 1, j]) \
                        or not color_same(mask_array[i, j], mask_array[i, j - 1]) \
                        or not color_same(mask_array[i, j], mask_array[i, j + 1]):
                    boundary_mask[i, j] = 255
    boundary_image = Image.fromarray(np.uint8(boundary_mask))
    boundary_image.save("./not.jpg")

    return boundary_image


def color_same(p1, p2):
    return p1[0] == p2[0] and p1[1] == p2[1] and p1[2] == p2[2]


def main():
    dump_root = Path(args.dump_boundary_root)
    dump_root.mkdir_p()

    for dr in Path(args.dataset_dir).dirs():

        files = sorted(dr.files())

        print(dr.name)
        Path(dump_root/dr.name).mkdir_p()
        for n, f in enumerate(files):
            image = boundary(f)
            print(f.name)
            image.save(os.path.join(dump_root, dr.name, f.name))


if __name__ == '__main__':
    main()
