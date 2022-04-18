import cv2
import os

import numpy as np
import pytesseract
from pytesseract import Output

# ниже путь к тессеракту(может быть другим)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class ImageTextExtractor:
    """Использую три метода выделения контура
           1 - с помощью pytesseract
           2 - opencv, выделяю все контуру которые есть на картинке
           3 - opencv, рисую прямоугольнику при обнаружении контура
    """

    # функция обработки изображения и изъятия мз него текста
    @staticmethod
    def process_opencv(filename):
        img = cv2.imread(filename)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # решение opencv(2)
        contours, hierarchy = cv2.findContours(gray_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        img1 = cv2.drawContours(gray_image, contours, -1, (0, 255, 75), 2)

        # решение opencv(3)
        for idx, contour in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(contour)
            # print("R", idx, x, y, w, h, cv2.contourArea(contour), hierarchy[0][idx])
            # hierarchy[i][0]: the index of the next contour of the same level
            # hierarchy[i][1]: the index of the previous contour of the same level
            # hierarchy[i][2]: the index of the first child
            # hierarchy[i][3]: the index of the parent
            if hierarchy[0][idx][3] == 0:
                cv2.rectangle(gray_image, (x, y), (x + w, y + h), (255, 0, 0), 1)

    @staticmethod
    def mask_method(img):
        # конвертируем цветовую палитру нашего изображения для дальнейшей обработки
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        """Сегментация с помощью маски
        Суть заключается в том, что мы берем определенный диапазон пикселей и "отсекаем" из изображения"""
        lower_white = np.array([0, 0, 0], dtype=np.uint8)
        upper_white = np.array([0, 0, 255], dtype=np.uint8)
        pix_from = (0, 0, 0)
        pix_to = (165, 165, 165)
        # фильтруем изображение
        mask = cv2.inRange(hsv_img, pix_from, pix_to)
        result = cv2.bitwise_and(img, img, mask=mask)
        mask1 = cv2.inRange(hsv_img, lower_white, upper_white)
        result1 = cv2.bitwise_and(img, img, mask=mask1)
        cv2.imshow("result1", result1)
        cv2.waitKey(0)

    @staticmethod
    def process_tesseract(filename):
        img = cv2.imread(filename)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray_image, 215, 255, cv2.THRESH_BINARY)

        height, width, _ = img.shape
        # рисуем прямоугольник не задевая нужный нам текст
        output = cv2.rectangle(thresh, (int(width / 4), int(height / 5)), (int(width * 4 / 5), int(height * 6 / 7)),
                               (0, 0, 0), -1)

        # configuring parameters for tesseract
        # oem - ocr engine mode psm - page segmentation mode
        # в tessedit_char_whitelist помещаем символы которые нам нужны
        custom_config = '--psm 11 --user-patterns eng.user-patterns ' \
                        '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:.+-°'

        # now feeding image to tesseract
        details = pytesseract.image_to_data(output, output_type=Output.DICT, config=custom_config)

        """помещает текст в квадрат"""
        """
        total_boxes = len(details['text'])
        for sequence_number in range(total_boxes):
            if float(details['conf'][sequence_number]) < 30:
                (x, y, w, h) = (details['left'][sequence_number], details['top'][sequence_number],
                                details['width'][sequence_number],
                                details['height'][sequence_number])

                output = cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        """

        
        cv2.imshow("img", img)
        cv2.imshow("output", output)
        cv2.waitKey(0)

        return list(filter(lambda x: x != "", details['text']))


if __name__ == '__main__':
    """С помощью модуля os загружаем файлы из папки автоматически"""

    folders = os.listdir('ExtractTextFromPhoto')
    image_numbers = 'ExtractTextFromPhoto/{folder}'
    im_path = 'ExtractTextFromPhoto/{folder}/{im_name}'

    a = ImageTextExtractor()
    for folder in folders:
        images = os.listdir(image_numbers.format(folder=folder))
        for image in images:
            print(a.process_tesseract(im_path.format(folder=folder, im_name=image)))
