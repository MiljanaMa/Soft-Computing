import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import SGD

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    ret, image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin

def invert(image):
    return 255-image

def display_image(image, color=False):
    if color:
        plt.imshow(image)
        plt.show()
    else:
        plt.imshow(image, 'gray')
        plt.show()

def dilate(image):
    kernel = np.ones((3, 3))
    return cv2.dilate(image, kernel, iterations=1)

def erode(image):
    kernel = np.ones((3, 3))
    return cv2.erode(image, kernel, iterations=1)

def resize_region(region):
    return cv2.resize(region, (28, 28), interpolation=cv2.INTER_NEAREST)

def select_roi(image_orig, image_bin):

    contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    regions_array = []
    connected_letters = []
    connect = False

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if  x > 280 and x < 800 and y > 170 and y < 280:

            if y > 200 and (h < 18 or (w < 20 and h < 30)):
                continue

            region = image_bin[y:y+h+1, x:x+w+1]
            regions_array.append([resize_region(region), (x, y, w, h)])
            
    regions_array = sorted(regions_array, key=lambda x: x[1][0])

    for i in range(len(regions_array)):

        if connect == True:
            connect = False
            continue

        current_region, current_rect = regions_array[i]
        if i == (len(regions_array) - 1):
            connected_letters.append([current_region, current_rect])
            break

        next_region, next_rect = regions_array[i + 1]
        if  (current_rect[0] + current_rect[2]) >= next_rect[0]:
            connect = True
            if (next_rect[0] + next_rect[2]) < (current_rect[0] + current_rect[2]):
                new_rect = (current_rect[0], next_rect[1], current_rect[2], current_rect[1] + current_rect[3] - next_rect[1])
                new_region = image_bin[new_rect[1]:new_rect[1]+new_rect[3]+1, new_rect[0]:new_rect[0]+new_rect[2]+1]
                connected_letters.append([resize_region(new_region), new_rect])
            else:
                new_rect = (current_rect[0], next_rect[1], (next_rect[0] + next_rect[2] - current_rect[0]), current_rect[1] + current_rect[3] - next_rect[1])
                new_region = image_bin[new_rect[1]:new_rect[1]+new_rect[3]+1, new_rect[0]:new_rect[0]+new_rect[2]+1]
                connected_letters.append([resize_region(new_region), new_rect])

        else:
            connected_letters.append([current_region, current_rect])
    
    sorted_regions = [region[0] for region in connected_letters]
    sorted_rectangles = [region[1] for region in connected_letters]
    region_distances = []

    for index in range(0, len(sorted_rectangles) - 1):
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index + 1]
        distance = next_rect[0] - (current[0] + current[2])
        region_distances.append(distance)
    
    return image_orig, sorted_regions, region_distances

#stackoverflow idea for comapring images
def image_similarity(img1, img2):
    diff_pixels = np.count_nonzero(np.bitwise_xor(img1, img2))
    total_pixels = np.prod(img1.shape)
    similarity_percentage = (total_pixels - diff_pixels) / total_pixels * 100
    return similarity_percentage

def is_similar(letter1, letter2, threshold=98.5):
    return image_similarity(letter1, letter2) >= threshold

def remove_duplicate(all_letters, threshold):
    new_unique_letters = []
    index = 0

    while index < len(all_letters):
        new_letter = all_letters[index]
        is_unique = True
        inner_index = 0

        while inner_index < len(new_unique_letters):
            if is_similar(new_letter, new_unique_letters[inner_index], threshold):
                is_unique = False
                break

            inner_index += 1

        if is_unique:
            new_unique_letters.append(new_letter)

        index += 1
    return new_unique_letters

def scale_to_range(image):
    return image/255

def matrix_to_vector(image):
    return image.flatten()

def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        scale = scale_to_range(region)
        ready_for_ann.append(matrix_to_vector(scale))
    return ready_for_ann

def convert_output(alphabet):
    nn_outputs = []
    for index in range(len(alphabet)):
        output = np.zeros(len(alphabet))
        output[index] = 1
        nn_outputs.append(output)
    return np.array(nn_outputs)

def create_ann(output_size):
    ann = Sequential()
    ann.add(Dense(128, input_dim=784, activation='relu'))
    ann.add(Dense(output_size, activation='softmax'))
    return ann

def train_ann(ann, X_train, y_train, epochs):
    X_train = np.array(X_train, np.float32)
    y_train = np.array(y_train, np.float32)
    
    sgd = SGD(learning_rate=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)
    ann.fit(X_train, y_train, epochs=epochs, batch_size=1, verbose=0, shuffle=False)
    return ann

def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]

def display_result_with_spaces(outputs, alphabet, distances):
    result = alphabet[winner(outputs[0])]
    for idx, output in enumerate(outputs[1:, :]):
        if distances[idx] > 15:
            result += ' '
        result += alphabet[winner(output)]
    return result

def hamming_distance(str1, str2):
    len_str1, len_str2 = len(str1), len(str2)

    if len_str1 == len_str2:
        return sum(c1 != c2 for c1, c2 in zip(str1, str2))
    else:
        min_len = min(len_str1, len_str2)
        distance = sum(c1 != c2 for c1, c2 in zip(str1[:min_len], str2[:min_len]))
        distance += abs(len_str1 - len_str2)
        return distance

data_path = sys.argv[1]
captcha_path = data_path + 'pictures/'

captcha = {
    "captcha_1.jpg": "klep klep",
    "captcha_2.jpg": "kto to je",
    "captcha_3.jpg": "švestka je",
    "captcha_4.jpg": "zemědělec",
    "captcha_5.jpg": "hlášení mačka",
    "captcha_6.jpg": "šťáva číšník",
    "captcha_7.jpg": "malý vlčiak",
    "captcha_8.jpg": "modré džínsy",
    "captcha_9.jpg": "hlavný baj",
    "captcha_10.jpg": "neúčastníckosť",
}

heming_sum = 0
alphabet = ['k', 'l', 'e', 'p', 't', 'o', 'j', 'š', 'v', 's', 'a', 'z', 'm', 'ě', 'd', 'c', 'h', 'á', 'n', 'í', 'č', 'ť', 'ý', 'i', 'r', 'é', 'ž', 'y', 'b', 'ú']

keras.utils.disable_interactive_logging()

all_letters = []
for captcha_filename, real_word in captcha.items():
    image_color = load_image(captcha_path + captcha_filename)
    img = image_bin(cv2.GaussianBlur(image_gray(image_color), (5, 5), 0))
    img_bin = erode(dilate(img))
    
    selected_regions, letters, region_distances = select_roi(image_color.copy(), img_bin)
    all_letters.extend(letters)
unique_letters = remove_duplicate(all_letters, 97.7)

inputs = prepare_for_ann(unique_letters)
outputs = convert_output(alphabet)
ann = create_ann(output_size=30)
ann = train_ann(ann, inputs, outputs, epochs=2000)

for captcha_filename, real_word in captcha.items():
    image_color = load_image(captcha_path + captcha_filename)
    img = image_bin(cv2.GaussianBlur(image_gray(image_color), (5, 5), 0))
    img_bin = erode(dilate(img))
    
    selected_regions, letters, distances = select_roi(image_color.copy(), img_bin)

    inputs = prepare_for_ann(letters)
    results = ann.predict(np.array(inputs, np.float32))
    predicted_word = display_result_with_spaces(results, alphabet, distances)
    print(captcha_filename + "-" + real_word + "-" + predicted_word)
    heming_sum += hamming_distance(real_word, predicted_word)
    
print("Hamming's distance is: " + str(heming_sum))
