import tensorflow as tf
import cv2
import imutils
import numpy as np
from imutils.contours import sort_contours
import argparse

from PIL import Image, ImageDraw, ImageFont
import PIL
import matplotlib.pyplot as plt
import os
import shutil
from numpy.core.records import array
from numpy.core.shape_base import block
import time
import sys

def img_y_shadow(img_b):
    (h,w)=img_b.shape
    a=[0 for z in range(0,h)]
    for i in range(0,h):          
        for j in range(0,w):      
            if img_b[i,j]==255:     
                a[i]+=1  
    return a

def img_x_shadow(img_b):
    (h,w)=img_b.shape
    a=[0 for z in range(0,w)]
    for i in range(0,h):          
        for j in range(0,w):      
            if img_b[i,j]==255:     
                a[j]+=1  
    return a

def img_show_array(a):
    plt.imshow(a)
    plt.show()

def show_shadow(arr, direction = 'x'):
    a_max = max(arr)
    if direction == 'x': # x轴方向的投影
        a_shadow = np.zeros((a_max, len(arr)), dtype=int)
        for i in range(0,len(arr)):
            if arr[i] == 0:
                continue
            for j in range(0, arr[i]):
                a_shadow[j][i] = 255
    elif direction == 'y': # y轴方向的投影
        a_shadow = np.zeros((len(arr),a_max), dtype=int)
        for i in range(0,len(arr)):
            if arr[i] == 0:
                continue
            for j in range(0, arr[i]):
                a_shadow[i][j] = 255

    img_show_array(a_shadow)

def get_shadowing_windows(img_path):
    img_bin = cv2.imread(img_path, 0) 
    thresh = 110
    ret,img_b=cv2.threshold(img_bin, thresh, 255, cv2.THRESH_BINARY_INV)

    # Shadowing
    img_x_shadow_a = img_x_shadow(img_b)
    img_y_shadow_a = img_y_shadow(img_b)
    
    y_s,y_t=0,0
    for i in range(len(img_y_shadow_a)):
        if img_y_shadow_a[i]>0:
            y_s=i
            break
    
    for i in range(len(img_y_shadow_a)):
        if img_y_shadow_a[len(img_y_shadow_a)-1-i]>0:
            y_t = len(img_y_shadow_a)-1-i
            break
    
    X = img_x_shadow_a 
    windows = []
    wd = [-1,-1]
    s=False
    for i in range(len(X)):
        if X[i] == 0 and not s:
            continue
        if X[i] == 0 and s:
            wd[1]=i
            windows.append(wd)
            wd=[-1,-1]
            s = False
        if X[i] > 0 and not s:
            s = True
            wd[0]=i
        if X[i] > 0 and s:
            continue
    return windows

def win_len(win):
    return win[1]-win[0]

def generate_char_windows_v1(img_path):
    windows = get_shadowing_windows(img_path)
    wdx_lens = [w[1]-w[0] for w in windows]
    mean_len = int(sum(wdx_lens) / len(wdx_lens)) # Here is the auto threshold, could be further adaptive.

    sorted_windows = sorted(windows, key=lambda x: x[0])
    merged = []
    to_be_merged = []
    current_window = sorted_windows[0]  
    for window in sorted_windows:
        if win_len(window) < mean_len:
            to_be_merged.append(window)
        else:
            if len(to_be_merged) > 0: 
                current_window = [to_be_merged[0][0], to_be_merged[-1][1]]
                merged.append(current_window)
                print('merge: ', [win_len(x) for x in to_be_merged], '->', win_len(current_window))
                to_be_merged = []
                merged.append(window)
            else:
                merged.append(window)
    if len(to_be_merged) > 0:  
        current_window = [to_be_merged[0][0], to_be_merged[-1][1]]
        merged.append(current_window)
        print('merge: ', [win_len(x) for x in to_be_merged], '->', win_len(current_window))
    return merged

def pad_char_img(original_image):
    orig_height, orig_width, _ = original_image.shape
    desired_size = max(orig_height, orig_width)+1
    # Calculate the padding needed for both dimensions
    pad_height = max(desired_size - orig_height, 0) // 2
    pad_width = max(desired_size - orig_width, 0) // 2
    
    # Pad the image with zeros (black pixels) to center it
    padded_image = cv2.copyMakeBorder(original_image, pad_height, pad_height, pad_width, pad_width,
                                       cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return padded_image
def is_overlap(rect1, rect2):
    # Extract coordinates and dimensions
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    # Check if the rectangles overlap in both x and y directions
    res = (x1 < x2 + w2) and (x1 + w1 > x2) and (y1 < y2 + h2) and (y1 + h1 > y2)
    return res

def find_covering_rectangle(rect1, rect2):
    # Extract coordinates and dimensions
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    # Calculate new rectangle coordinates
    new_x = min(x1, x2)
    new_y = min(y1, y2)
    new_w = max(x1 + w1, x2 + w2) - new_x
    new_h = max(y1 + h1, y2 + h2) - new_y

    return new_x, new_y, new_w, new_h

def merge_overlapping_rectangles(rectangles):
    # Sort rectangles by x-coordinate
    sorted_rectangles = sorted(rectangles, key=lambda rect: rect[0])

    merged_rectangles = []
    processed = [False] * len(sorted_rectangles)

    for i in range(len(sorted_rectangles)):
        if not processed[i]:
            merged_rect = sorted_rectangles[i]
            for j in range(i + 1, len(sorted_rectangles)):
                if not processed[j] and is_overlap(merged_rect, sorted_rectangles[j]):
                    merged_rect = find_covering_rectangle(merged_rect, sorted_rectangles[j])
                    processed[j] = True
            merged_rectangles.append(merged_rect)
            processed[i] = True

    return merged_rectangles

def merge_A_symbol(windows):
    skips=[]
    aft_merge_special_symbol= []
    for i in range(len(windows)):
        if i in skips:
            continue
        wdx = windows[i]
        if i + 2 < len(windows):
            wdx1, wdx2 = windows[i+1], windows[i+2]
            
            x11, x12 = wdx1[0], wdx1[0]+wdx1[2]
            x21, x22 = wdx2[0], wdx2[0]+wdx2[2]
            if x12 > x21:
                min_x = min(wdx[0], wdx1[0], wdx2[0])
                min_y = min(wdx[1], wdx1[1], wdx2[1])
                max_x = max(wdx[0]+wdx[2], wdx1[0]+wdx1[2], wdx2[0]+wdx2[2])
                max_y = max(wdx[1]+wdx[3], wdx1[1]+wdx1[3], wdx2[1]+wdx2[3])
                aft_merge_special_symbol.append((min_x, min_y, max_x - min_x, max_y - min_y))
                skips += [i+1, i+2]
            else:
                aft_merge_special_symbol.append(wdx)
        else:
            aft_merge_special_symbol.append(wdx)
    return aft_merge_special_symbol

def generate_char_windows_v2(img_path):
    myImage = cv2.imread(img_path)
    grayImg = cv2.cvtColor(myImage, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(grayImg, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
    dilation = cv2.dilate(thresh1, horizontal_kernel, iterations=1)
    horizontal_contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    im2 = myImage.copy()
    for cnt in horizontal_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (255, 255, 255), 0)
    windows = seg_word(rect)
    return [(win[0], win[0]+win[2]) for win in windows]

def seg_word(wordImage):
    # ref: https://github.com/sunaarun/ocr_segmentation
    # convert the input image int gray scale
    grayImg = cv2.cvtColor(wordImage, cv2.COLOR_BGR2GRAY)
    # Binarize the gray image with OTSU algorithm
    ret, thresh2 = cv2.threshold(grayImg, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    # create a Structuring Element size of 8*10 for the vertical contouring
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # apply Dilation for once only
    dilation = cv2.dilate(thresh2, vertical_kernel, iterations=1)
    # fingd the vertical contours
    vertical_contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # Run through each contour and extract the bounding box
    windows = []
    for cnt in vertical_contours:
        #computes the minimum rectangle
        x, y, w, h = cv2.boundingRect(cnt)
        windows.append((x, y, w, h))
    
    windows = merge_overlapping_rectangles(windows)
    windows = merge_A_symbol(windows) # Specific merge for Symbol-transit.
    ## Below code is for visualization.
    # word_img = wordImage.copy()
    # for i in range(len(windows)):
    #     (x, y, w, h) = windows[i]
    #     rect = cv2.rectangle(word_img, (x, y), (x + w, y + h), (0, 255, 0), 0)
    
    return windows

if __name__=="__main__":
    # process input
    # Load image
    img_path = './errors/'+sys.argv[1]
    img_raw =cv2.imread(img_path) 

    print("Use Simple shadowing method for segementation! ")
    merged_windows = generate_char_windows_v1(img_path)

    # print("Use Image morphology method for segementation! ")
    # merged_windows = generate_char_windows_v2(img_path)

    # Segementing
    final_imgs = []
    for i in range(len(merged_windows)):
        left, right = merged_windows[i][0], merged_windows[i][1]
        img = img_raw[:,left:right]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret,img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        final_imgs.append(img)
    
    # Load model
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D']
    TF_MODEL_FILE_PATH = 'model_v2.tflite' # The default path to the saved TensorFlow Lite model
    
    interpreter = tf.lite.Interpreter(model_path=TF_MODEL_FILE_PATH)
    classify_lite = interpreter.get_signature_runner('serving_default')

    # Run model
    final_res=[]
    for i in range(len(final_imgs)):
        padded_image = pad_char_img(final_imgs[i]) # This line is important, do keep it.
        
        # You can resize the img to 32X32 using other ways, these 2 are not important.
        im1 = Image.fromarray(padded_image)
        im1 = im1.resize((32, 32))
        
        # Here I create a batch of size 1 to OCR char one by one,
        # You can make batch size larger than 1 to perform bulk OCR.
        img_array = tf.keras.utils.img_to_array(im1)
        img_array = tf.expand_dims(img_array, 0) # Create a batch
        
        predictions_lite = classify_lite(rescaling_input=img_array)['dense_1']
        score_lite = tf.nn.softmax(predictions_lite)
        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score_lite)], 100 * np.max(score_lite))
        )
        final_res.append(class_names[np.argmax(score_lite)])
    print('Final recognized result, len:',len(final_res))
    print(''.join(final_res))