import cv2

class ImageTextExtractor:
    def Process(self,filename):
        img=cv2.imread(filename,-1)
        cv2.imshow('girl', img)
        print("Высота:" + str(img.shape[0]))
        print("Ширина:" + str(img.shape[1]))

        cv2.waitKey(0)



if __name__ == '__main__':
    a=ImageTextExtractor()
    a.Process('ExtractTextFromPhoto/Video 1 with text/Image 2021-10-23 16-44-15-344.pgm')