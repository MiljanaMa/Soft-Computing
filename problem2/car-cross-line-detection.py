import os
import sys
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)

def detect_vertical_red_line(img, frameNum):

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    red_mask = cv2.inRange(hsv_img, lower_red, upper_red)

    
    edges_img = cv2.Canny(red_mask, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(image=edges_img, rho=1, theta=np.pi/180, threshold=20, lines=np.array([]),
                            minLineLength=5, maxLineGap=50)

    if lines is not None and len(lines) > 0:
        x1 = lines[0][0][0]
        y1 = lines[0][0][1]
        x2 = lines[0][0][2]
        y2 = lines[0][0][3]
        return (x1, y1, x2, y2)
    else:
        return None

def get_hog():
    img_size = (60, 120)
    nbins = 9
    cell_size = (8, 8)
    block_size = (2, 2)
    hog = cv2.HOGDescriptor(_winSize=(img_size[1] // cell_size[1] * cell_size[1],
                                      img_size[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)
    return hog

def classify_window(window, clf_svm, hog):
    features = hog.compute(window).reshape(1, -1)
    return clf_svm.predict_proba(features)[0][1]

def jaccard_index_similar(first_box, second_box, window_size):  

    x1, y1 = first_box
    x2, y2 = second_box    
    intersection = max(0, min(x1 + window_size[0], x2 + window_size[0]) - max(x1, x2)) * max(0, min(y1 + window_size[1], y2 + window_size[1]) - max(y1, y2))  
    area = window_size[0] * window_size[1]
    iou = intersection / float(2 * area - intersection)
    
    return iou

def detect_cars(image, step, clf_svm, hog, window_size=(160, 76)):
    all_windows = []
    real_detections = []
    for y in range(0, image.shape[0], step):
        for x in range(0, image.shape[1], step):
            this_window = (x, y) 
            window = image[y:y + window_size[1], x:x + window_size[0]]
            window = cv2.resize(window, (120, 60), interpolation=cv2.INTER_NEAREST)
            score = classify_window(window, clf_svm, hog)
            if(score > 0.85):
                all_windows.append((score, this_window))

    all_windows.sort(key=lambda x: x[0], reverse=True)
    while len(all_windows) > 0:
        _, max_scored_window = all_windows[0]
        real_detections.append(max_scored_window)

        all_windows = all_windows[1:]
        left_detections = []
        for detection in all_windows:
            iou = jaccard_index_similar(detection[1], max_scored_window, window_size)
            if iou < 0.3:
                left_detections.append(detection)

        all_windows = left_detections

    indexes = []
    for i in range(len(real_detections)):
        x1, y1 = real_detections[i]
        j = i + 1
        while j < len(real_detections):
            x2, y2 = real_detections[j]

            if abs(y1 - y2) < 15 and abs(x2 - (x1 + 160)) < 90:
                indexes.append(i)
                indexes.append(j)
            j += 1

    for i in sorted(indexes, reverse=True):
        del real_detections[i]

    return real_detections
   
def process_video(video_path, clf_svm, hog, target_resolution=(1080, 607)):
    sum_of_nums = 0
    frame_num = 0
    cap = cv2.VideoCapture(video_path)
    cap.set(1, frame_num)
    
    while True:
        frame_num += 1    
        grabbed, frame = cap.read()

        if not grabbed:
            break
    
        frame = cv2.resize(frame, target_resolution, interpolation=cv2.INTER_LINEAR)
        if frame_num % 4 == 0:
            line_coords = detect_vertical_red_line(frame, frame_num) 
            coord_x = line_coords[0]
            bottom_y = line_coords[1]

            rectangles = detect_cars(frame, 10, clf_svm, hog)
            for rectangle in rectangles:
                
                if (bottom_y > rectangle[1]) and (rectangle[0] <= coord_x <= rectangle[0] + 160):
                    window_size=(190, 90)
                    cv2.rectangle(frame, (rectangle[0], rectangle[1]), (rectangle[0] + window_size[0], rectangle[1] + window_size[1]), (0, 255, 0), 2)
                    plt.imshow(frame, 'gray')
                    plt.show()
                    sum_of_nums += 1
    cap.release()
    return sum_of_nums

data_path = sys.argv[1]
train_dir = data_path +'pictures/'

pos_imgs = []
neg_imgs = []

for img_name in os.listdir(train_dir):
    img_path = os.path.join(train_dir, img_name)
    img = load_image(img_path)
    if 'p_' in img_name:
        pos_imgs.append(img)
    elif 'n_' in img_name:
        neg_imgs.append(img)

pos_features = []
neg_features = []
labels = []

hog = get_hog()

for img in pos_imgs:
    pos_features.append(hog.compute(img))
    labels.append(1)

for img in neg_imgs:
    neg_features.append(hog.compute(img))
    labels.append(0)

x = np.vstack((np.array(pos_features), np.array(neg_features)))
y = np.array(labels)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

clf_svm = SVC(kernel='linear', probability=True)
clf_svm.fit(x_train, y_train)

videosPath = data_path + 'videos/'
videoCounts = {
    "segment_1.mp4": 4,
    "segment_2.mp4": 12,
    "segment_3.mp4": 4,
    "segment_4.mp4": 11,
    "segment_5.mp4": 3,
}
sum = 0
examples = 0
for videoFilename, realCount in videoCounts.items():
    examples += 1
    predictedCount = process_video(videosPath + videoFilename, clf_svm, hog)
    sum += abs(realCount - predictedCount)
    print(videoFilename + '-' + str(realCount) + '-' + str(predictedCount))

mae = sum/examples
print('Mean absolute error: ' + str(mae))