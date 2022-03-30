import numpy as np
import os
import matplotlib.pyplot as plt
from IPython.display import Image, display

import tensorflow as tf
from keras.preprocessing.image import load_img,img_to_array,ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Dense, Flatten, Activation
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNet
from tensorflow.python.client import device_lib


#Setup base directories for all of the data files in the ncku-bird-classification 
BASE_DIR = 'base_directory_path'
#print('BASE_DIR: ', os.listdir(BASE_DIR))
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TEST_DIR = os.path.join(BASE_DIR, 'test')


#This will get number of train groups categories
Category_count = len(os.listdir(TRAIN_DIR))
#print(Category_count)


#Load an image to determine image shape for analysis
IMAGE = load_img('random_image_path')
#plt.imshow(IMAGE)
#plt.axis("off")
#plt.show()

IMAGEDATA = img_to_array(IMAGE)
SHAPE = IMAGEDATA.shape
#print('Figures are ', SHAPE)


#This will be used on training, test data
General_datagen = ImageDataGenerator(rescale=1./255, )

train_data = General_datagen.flow_from_directory(TRAIN_DIR, target_size=(224,224))
#print('data groups:', len(train_data)) #Will be used to determine steps_per_epoch in model
Train_groups = len(train_data)

test_data = General_datagen.flow_from_directory(TEST_DIR, target_size=(224,224),)
#print('data groups:', len(test_data)) #Will be used to determine steps_per_epoch in model
Test_groups = len(test_data)


#Applied Data augmentation to training data.  
Augment_datagen = ImageDataGenerator(rescale=1./255, 
    rotation_range=40, 
    width_shift_range=0.2, 
    height_shift_range=0.2,
    zoom_range=0.2, 
    horizontal_flip=True, 
    fill_mode='nearest') 
Augmentation_train = Augment_datagen.flow_from_directory(TRAIN_DIR, target_size=(224,224))
#print('data groups:', len(Augmentation_train)) #same as Train_groups


#Bring in the imagenet dataset training weights for the Mobilenet CNN model
base_mobilenet = MobileNet(weights = 'imagenet', include_top = False, 
                           input_shape = SHAPE)
base_mobilenet.trainable = False # Freeze the mobilenet weights

model = Sequential()
model.add(base_mobilenet)

model.add(Flatten()) 
model.add(Activation('relu'))
model.add(Dense(Category_count)) 
model.add(Activation('softmax'))

model.summary()

#Compile
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
               loss = 'categorical_crossentropy',
               metrics = ['accuracy'])
#fit model
history = model.fit( 
    train_data, 
    steps_per_epoch = Train_groups, 
    epochs = 12,
    validation_data = test_data,
    validation_steps = Test_groups,
    verbose = 1,
    callbacks=[EarlyStopping(monitor = 'val_accuracy', patience = 5, 
                             restore_best_weights = True),
               ReduceLROnPlateau(monitor = 'val_loss', factor = 0.7, 
                                 patience = 2, verbose = 1)]) 
                # let verbose = 1 can see the learning rate decay


#plot accuracy vs epoch
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot loss values vs epoch
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Evaluate against test data.
scores = model.evaluate(test_data, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

model.save('my_model.h5')

