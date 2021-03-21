import cv2

def findSpeechBubbles(imagePath, method = 'not_simple'):
    image = cv2.imread(imagePath)
    hight, width, _ = image.shape
    # print("hight: {}, width: {}".format(hight, width))
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageGrayBlur = cv2.GaussianBlur(imageGray,(3,3),0)

    if method != 'simple':
        imageGrayBlurCanny = cv2.Canny(imageGrayBlur,50,500)
        binary = cv2.threshold(imageGrayBlurCanny,235,255,cv2.THRESH_BINARY)[1]
    else:
        binary = cv2.threshold(imageGrayBlur,235,255,cv2.THRESH_BINARY)[1]  
    contours, _ = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    croppedImageList = []
    for contour in contours:
        rect = cv2.boundingRect(contour)
        [x, y, w, h] = rect
        print("x: {}, y: {}, w: {}, h: {}".format(x, y, w, h))
        if w < 500 and w > 60 and h < 500 and h > 60:
            if h / hight < 0.95:
                croppedImage = image[y:y+h, x:x+w]
                croppedImageList.append(croppedImage)

    return croppedImageList

if __name__ == '__main__':
    list_crop_image = findSpeechBubbles('data/4_48_4.jpg')
    print(len(list_crop_image))
    for image in list_crop_image:
        cv2.imshow("image", image)
        cv2.waitKey(0)