import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt 

def process_image(image_filename):
    
    img = cv2.imread(image_filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    
    img_crop = img[170:480, :]

    
    img_gray = cv2.cvtColor(img_crop, cv2.COLOR_RGB2GRAY)

   
    img_bin = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 55, 10)

    
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    img_bin = cv2.erode(img_bin, kernel, iterations=4)
    img_bin = cv2.dilate(img_bin, kernel, iterations=3)

    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    contours_pok = []
    for contour in contours:
        center, radius = cv2.minEnclosingCircle(contour)
        center, size, angle = cv2.minAreaRect(contour)
        width, height = size

        if 20 < radius < 64 and 50 < width < 165 and 56 < height < 105:
            contours_pok.append(contour)

    return len(contours_pok)

folderPath = sys.argv[len(sys.argv) - 1]
imageCounts = {
    "picture_1.jpg": 3,
    "picture_2.jpg": 9,
    "picture_3.jpg": 6,
    "picture_4.jpg": 8,
    "picture_5.jpg": 10,
    "picture_6.jpg": 2,
    "picture_7.jpg": 5,
    "picture_8.jpg": 4,
    "picture_9.jpg": 4,
    "picture_10.jpg": 10,

}

sum = 0
examples = 0
for imageFilename, realCount in imageCounts.items():
    examples += 1
    predictedCount = process_image(folderPath + imageFilename)
    sum += abs(realCount - predictedCount)
    print(imageFilename + '-' + str(realCount) + '-' + str(predictedCount))

mae = sum/examples
print('Mean absolute error: ' + str(mae))


