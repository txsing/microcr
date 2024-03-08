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

def generate_windows(X):
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

def merge_windows(windows, gap):
    sorted_windows = sorted(windows, key=lambda x: x[0])
    
    merged = []  
    current_window = sorted_windows[0]  
    
    for window in sorted_windows[1:]:
        if current_window[1] >= window[0] - gap:
            current_window[1] = max(current_window[1], window[1])
        else:
            merged.append(current_window)
            current_window = window
    
    merged.append(current_window)
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

if __name__=="__main__":
    # process input
    # Load image
    img_path = './m1.jpg'
    img_raw =cv2.imread(img_path) 
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

    windows = generate_windows(img_x_shadow_a)
    merged_windows = merge_windows(windows.copy(), 7)
    lengths = [t[1]-t[0] for t in merged_windows]
    max_wdw_len=max(lengths)
    
    # Segementing
    final_imgs = []
    for i in range(len(merged_windows)):
        left, right = merged_windows[i][0], merged_windows[i][1]
        pad = int((max_wdw_len-(right-left))/2)
        img = img_raw[:,left-pad:right+pad]
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
        padded_image = pad_char_img(final_imgs[i])
        im1 = Image.fromarray(padded_image)
        im1 = im1.resize((32, 32))
        
        img_array = tf.keras.utils.img_to_array(im1)
        img_array = tf.expand_dims(img_array, 0) # Create a batch
        
        predictions_lite = classify_lite(rescaling_input=img_array)['dense_1']
        score_lite = tf.nn.softmax(predictions_lite)
        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score_lite)], 100 * np.max(score_lite))
        )
        final_res.append(class_names[np.argmax(score_lite)])
    print('Final recognized result')
    print(final_res)