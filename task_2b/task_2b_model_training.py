############################## FILL THE MANDATORY INFORMATION BELOW ###############################

# Team ID:			[ GG_2956 ]
# Author List:		[ Saloni Gavde, Atharva Magre, Harshad Joshi ]
# Filename:			task_2b_model_training.py
###################################################################################################

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
import os
import cv2


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)

directory = 'training'

#loading images in dataset
data = tf.keras.utils.image_dataset_from_directory('training')
#scaling data
data = data.map(lambda x,y: (x/255,y))
#generating iterator
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()


#defining training and validation size
train_size = int(len(data)*0.75)+1
val_size = int(len(data)*0.25)

#peforming the train-validation split
train = data.take(train_size)
val = data.skip(train_size).take(val_size)

#model architecture
model = Sequential() #instantiate model
model.add(Conv2D(128,(3,3),1,activation='relu',input_shape=(256,256,3)))
model.add(MaxPooling2D())

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256,activation='relu'))
model.add(Dense(5,activation='softmax'))

#define optimizer and loss function
model.compile('adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#train the model
model.fit(train, epochs=15, validation_data=val)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")