import cv2
import os

class ImageTextExtractor:
    def Process(self,filename):
        img=cv2.imread(filename,-1)
        cv2.imshow('im', img)
        print(filename)
        cv2.waitKey(0)



if __name__ == '__main__':
    folders=os.listdir('ExtractTextFromPhoto')
    image_numbers = 'ExtractTextFromPhoto/{folder}'
    im_path='ExtractTextFromPhoto/{folder}/{im_name}'
    a=ImageTextExtractor()
    for folder in folders:
        images=os.listdir(image_numbers.format(folder=folder))
        for image in images:
            a.Process(im_path.format(folder=folder,im_name=image))
