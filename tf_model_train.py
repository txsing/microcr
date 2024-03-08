import os

import numpy as np

import tensorflow as tf
from tensorflow import keras 
from keras import layers 
assert tf.__version__.startswith('2')

def load_data(data_dir):
    train_ds = tf.keras.utils.image_dataset_from_directory(
      data_dir,
      validation_split=0.2,
      subset="training",
      seed=123,
      image_size=(32, 32),
      batch_size=128)
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
      data_dir,
      validation_split=0.2,
      subset="validation",
      seed=123, # keep the seed the same as in train
      image_size=(32, 32),
      batch_size=128)
    return train_ds, val_ds

def define_model():
    # model = tf.keras.models.Sequential([ 
    #     layers.Conv2D(16, (3, 3), activation='relu', 
    #                   input_shape=(32, 32, 3), padding='same'), 
    #     layers.Conv2D(32, (3, 3), 
    #                   activation='relu', 
    #                   padding='same'), 
    #     layers.Conv2D(64, (3, 3), 
    #                   activation='relu', 
    #                   padding='same'), 
    #     layers.MaxPooling2D(2, 2), 
    #     layers.Conv2D(128, (3, 3), 
    #                   activation='relu', 
    #                   padding='same'), 
    #     layers.Flatten(), 
    #     layers.Dense(256, activation='relu'), 
    #     layers.BatchNormalization(), 
    #     layers.Dense(256, activation='relu'), 
    #     layers.Dropout(0.3), 
    #     layers.BatchNormalization(), 
    #     layers.Dense(14, activation='softmax') 
    # ]) 
    
    
    # model = tf.keras.Sequential([
    #   tf.keras.layers.Rescaling(1./255),
    #   tf.keras.layers.Conv2D(32, 3, activation='relu'),
    #   tf.keras.layers.MaxPooling2D(),
    #   tf.keras.layers.Conv2D(32, 3, activation='relu'),
    #   tf.keras.layers.MaxPooling2D(),
    #   tf.keras.layers.Conv2D(32, 3, activation='relu'),
    #   tf.keras.layers.MaxPooling2D(),
    #   tf.keras.layers.Flatten(),
    #   tf.keras.layers.Dense(128, activation='relu'),
    #   tf.keras.layers.Dense(num_classes)
    # ])
    
    
    model = tf.keras.Sequential([
      tf.keras.layers.Rescaling(1./255),
      tf.keras.layers.Conv2D(16, 3, activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(64, 3, activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(num_classes)
    ])
    return model

def convert_2_tflite(src_tf_model, tflite_model_name):
    converter = tf.lite.TFLiteConverter.from_keras_model(src_tf_model)
    tflite_model = converter.convert()
    with open(tflite_model_name, 'wb') as f:
        f.write(tflite_model)
    file_stats = os.stat(tflite_model_name)
    print(f'Model Size in KB is {file_stats.st_size / (1024)}')


if __name__=="__main__":
    data_dir= './tmp/MICRSTTF'
    tflite_model_name = 'model_bw_test.tflite'
    
    train_ds, val_ds = load_data(data_dir)
    class_names = train_ds.class_names
    num_classes = len(class_names) # 14
    
    model = define_model()
    model.compile(
      optimizer='adam',
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'])
    
    model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=12
    )
    
    tflitemodel = convert_2_tflite(model, tflite_model_name)
    
    interpreter = tf.lite.Interpreter(model_path=f"./{tflite_model_name}")
    classify_lite = interpreter.get_signature_runner('serving_default')
    print(interpreter.get_signature_list())
    
    