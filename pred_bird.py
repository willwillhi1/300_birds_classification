import numpy as np
import os
import matplotlib.pyplot as plt
from IPython.display import Image, display

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img,img_to_array,ImageDataGenerator
from tensorflow.keras.utils import plot_model
import csv

#load model
model = tf.keras.models.load_model('model_path')

#check model load correctly
model.summary()

#get all class names
TRAIN_DIR = 'train_dataset_parh'
General_datagen = ImageDataGenerator(rescale=1./255, )
train_data = General_datagen.flow_from_directory(TRAIN_DIR, target_size=(224,224))
clsss_name = list(train_data.class_indices.keys())


path = 'grading_data_path'

#get number of images in grading_data
num_files = len([f for f in os.listdir(path)if os.path.isfile(os.path.join(path, f))])

result = []
for i in range(num_files):
    #load image
    image_path = os.path.join(path, str(i)+'.jpg')
    my_image = load_img(image_path, target_size=(224, 224))
    my_image = img_to_array(my_image)
    #data preprocessing
    my_image = my_image.astype('float32')/255
    #reshape image shape to (1,224,224,3)
    my_image = my_image.reshape((1, my_image.shape[0], my_image.shape[1], my_image.shape[2]))
    #prediction
    prediction = model.predict(my_image)
    #add prediction to result lsit
    result.append(clsss_name[np.argmax(prediction)])


#create write.csv and write result into result.csv
with open('result', 'w') as f:   
    # using csv.writer method from CSV package
    write = csv.writer(f)  
    write.writerow(result)
