import cv2
import os
import pytesseract
from pytesseract import Output

# ниже путь к тессеракту(может быть другим)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

"""
"""


class ImageTextExtractor:
    # функция обработки изображения и изъятия мз него текста
    @staticmethod
    def process(filename):
        img = cv2.imread(filename)

        # конвертируем цветовую палитру нашего изображения для дальнейшей обработки
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        """Сегментация с помощью маски
        Суть заключается в том, что мы берем определенный диапазон пикселей и "отсекаем" из изображения"""
        pix_from = (0, 0, 0)
        pix_to = (170, 170, 170)
        # фильтруем изображение
        mask = cv2.inRange(hsv_img, pix_from, pix_to)
        result = cv2.bitwise_and(img, img, mask=mask)
        # переводим изображение в серую палитру
        gray_image = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        """Использую три метода выделения контура
        1 - с помощью pytesseract
        2 - opencv, выделяю все контуру которые есть на картинке
        3 - opencv, рисую прямоугольнику при обнаружении контура
        """
        # создаем 3 копии для вывода результата обнаружения контура
        output = result.copy()
        output1 = result.copy()
        output2 = result.copy()

        # решение с тессерактом
        # configuring parameters for tesseract
        # oem - ocr engine mode psm - page segmentation mode
        custom_config = r'--oem 1 --psm 6'
        # now feeding image to tesseract
        details = pytesseract.image_to_data(result, output_type=Output.DICT, config=custom_config, lang='eng')
        total_boxes = len(details['text'])
        for sequence_number in range(total_boxes):
            if float(details['conf'][sequence_number]) > 30:
                (x, y, w, h) = (details['left'][sequence_number], details['top'][sequence_number],
                                details['width'][sequence_number],
                                details['height'][sequence_number])

                output = cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # решение opencv(2)
        contours, hierarchy = cv2.findContours(gray_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        img1 = cv2.drawContours(output1, contours, -1, (0, 255, 75), 2)

        # решение opencv(3)
        for idx, contour in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(contour)
            # print("R", idx, x, y, w, h, cv2.contourArea(contour), hierarchy[0][idx])
            # hierarchy[i][0]: the index of the next contour of the same level
            # hierarchy[i][1]: the index of the previous contour of the same level
            # hierarchy[i][2]: the index of the first child
            # hierarchy[i][3]: the index of the parent
            if hierarchy[0][idx][3] == 0:
                cv2.rectangle(output2, (x, y), (x + w, y + h), (255, 0, 0), 1)

        # расскоментировать, чтобы увидеть результат
        # cv2.imshow("output", output)
        # cv2.imshow("output1", output1)
        # cv2.imshow("output2", output2)
        # cv2.waitKey(0)

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
            print(a.process(im_path.format(folder=folder, im_name=image)))
