import math

import cv2
import os

import numpy as np
import pytesseract
from pytesseract import Output
from matplotlib import pyplot as plt
import sys
from PIL import Image

# ниже путь к тессеракту(может быть другим)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class ImageTextExtractor:
    from PIL import Image

    @staticmethod
    def process_tesseract(filename, q):
        img = cv2.imread(filename)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #gray_image = cv2.bilateralFilter(gray_image, 11, 17, 17)  # Blur to reduce noise
        ret, output = cv2.threshold(gray_image,150, 255, cv2.THRESH_BINARY_INV)
        # oem - ocr engine mode psm - page segmentation mode
        custom_config = '--psm 11 --oem 1 -l eng -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:.+-°'

        details = pytesseract.image_to_data(output, output_type=Output.DICT, config=custom_config)
        # cv2.imwrite(path.format(q=q), output)
        # print(path.format(q=q))
        q = q + 1
        print(list(filter(lambda x: x != "", details['text'])))
        cv2.imshow("img", img)
        cv2.imshow("output", output)
        cv2.waitKey(0)

        return list(filter(lambda x: x != "", details['text']))


def deinterlace_file(inp, output_format_str, row_names=('Left', 'Right')):
    print("Deinterlacing {}".format(inp))
    source = Image.open(inp)
    source.load()
    dim = source.size

    scaled_size1 = (math.floor(dim[0]), math.floor(dim[1] / 2) + 1)
    scaled_size2 = (math.floor(dim[0] / 2), math.floor(dim[1] / 2) + 1)

    top = Image.new(source.mode, scaled_size1)
    top_pixels = top.load()
    other = Image.new(source.mode, scaled_size1)
    other_pixels = other.load()
    for row in range(dim[1]):
        for col in range(dim[0]):
            pixel = source.getpixel((col, row))
            row_int = math.floor(row / 2)
            if row % 2:
                top_pixels[col, row_int] = pixel
            else:
                other_pixels[col, row_int] = pixel

    top_final = top.resize(scaled_size2, Image.NEAREST)  # Downsize to maintain aspect ratio
    other_final = other.resize(scaled_size2, Image.NEAREST)  # Downsize to maintain aspect ratio
    top_final.save(output_format_str.format(row_names[0]))
    other_final.save(output_format_str.format(row_names[1]))


if __name__ == '__main__':
    """С помощью модуля os загружаем файлы из папки автоматически"""
    folders = os.listdir('ExtractTextFromPhoto')
    image_numbers = 'ExtractTextFromPhoto/{folder}'
    im_path = 'ExtractTextFromPhoto/{folder}/{im_name}'
    q = 0
    a = ImageTextExtractor()
    f='ExtractTextFromPhoto/{folder}/file{q}.png'
    for folder in folders:
        images = os.listdir(image_numbers.format(folder=folder))
        for image in images:
            a.process_tesseract(im_path.format(folder=folder, im_name=image), q)
            #deinterlace_file(im_path.format(folder=folder, im_name=image),f.format(folder=folder,q=q) )
            q = q + 1
