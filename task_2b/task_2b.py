'''
*****************************************************************************************
*
*        		===============================================
*           		GeoGuide(GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement Task 2B of GeoGuide(GG) Theme (eYRC 2023-24).
*
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''
############################## FILL THE MANDATORY INFORMATION BELOW ###############################

# Team ID:			[ GG_2956 ]
# Author List:		[ Saloni Gavde, Atharva Magre, Harshad Joshi ]
# Filename:			task_2b.py
# Functions:	    [`classify_event(image)` ]
###################################################################################################

# IMPORTS (DO NOT CHANGE/REMOVE THESE IMPORTS)
from sys import platform
import numpy as np
import subprocess
import shutil
import ast
import sys
import os

# Additional Imports
'''
You can import your required libraries here
'''
import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json


# DECLARING VARIABLES (DO NOT CHANGE/REMOVE THESE VARIABLES)
detected_list = []
numbering_list = []
img_name_list = []

# Declaring Variables
'''
You can delare the necessary variables here
'''

# EVENT NAMES
'''
We have already specified the event names that you should train your model with.
DO NOT CHANGE THE BELOW EVENT NAMES IN ANY CASE
If you have accidently created a different name for the event, you can create another 
function to use the below shared event names wherever your event names are used.
(Remember, the 'classify_event()' should always return the predefined event names)  
'''
combat = "combat"
rehab = "humanitarianaid"
military_vehicles = "militaryvehicles"
fire = "fire"
destroyed_building = "destroyedbuilding"
###################################################################################################
###################################################################################################
''' 
    Purpose:
    ---
    This function will load your trained model and classify the event from an image which is 
    sent as an input.

    Input Arguments:
    ---
    `image`: Image path sent by input file 	

    Returns:
    ---
    `event` : [ String ]
                          Detected event is returned in the form of a string

    Example call:
    ---
    event = classify_event(image_path)
    '''


def classify_event(image):
    json_file = open('C:/Users/HP/PycharmProjects/pythonProject/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("C:/Users/HP/PycharmProjects/pythonProject/model.h5")

    # Preprocess the input image
    img = keras.preprocessing.image.load_img(image, target_size=(256, 256))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # Make predictions
    predictions = loaded_model.predict(img_array)

    # Get the class index with the highest probability
    predicted_class_index = np.argmax(predictions)

    # Map the index to the corresponding event name
    event_names = [combat, destroyed_building, fire, rehab, military_vehicles]
    detected_event = event_names[predicted_class_index]

    return detected_event


# ADDITIONAL FUNCTIONS
'''
Although not required but if there are any additonal functions that you're using, you shall add them here. 
'''


###################################################################################################
########################### DO NOT MAKE ANY CHANGES IN THE SCRIPT BELOW ###########################
def classification(img_name_list):
    for img_index in range(len(img_name_list)):
        img = "events/" + str(img_name_list[img_index]) + ".jpeg"
        detected_event = classify_event(img)
        print((img_index + 1), detected_event)
        if detected_event == combat:
            detected_list.append("combat")
        if detected_event == rehab:
            detected_list.append("rehab")
        if detected_event == military_vehicles:
            detected_list.append("militaryvehicles")
        if detected_event == fire:
            detected_list.append("fire")
        if detected_event == destroyed_building:
            detected_list.append("destroyedbuilding")
    shutil.rmtree('events')
    return detected_list


def detected_list_processing(detected_list):
    try:
        detected_events = open("detected_events.txt", "w")
        detected_events.writelines(str(detected_list))
    except Exception as e:
        print("Error: ", e)


def input_function():
    if platform == "win32":
        try:
            subprocess.run("input.exe")
        except Exception as e:
            print("Error: ", e)
    if platform == "linux":
        try:
            subprocess.run("./input")
        except Exception as e:
            print("Error: ", e)
    img_names = open("image_names.txt", "r")
    img_name_str = img_names.read()

    img_name_list = ast.literal_eval(img_name_str)
    return img_name_list


def output_function():
    if platform == "win32":
        try:
            subprocess.run("output.exe")
        except Exception as e:
            print("Error: ", e)
    if platform == "linux":
        try:
            subprocess.run("./output")
        except Exception as e:
            print("Error: ", e)


###################################################################################################
def main():
    ##### Input #####
    img_name_list = input_function()
    #################

    ##### Process #####
    detected_list = classification(img_name_list)
    detected_list_processing(detected_list)
    ###################

    ##### Output #####
    output_function()
    ##################


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        if os.path.exists('events'):
            shutil.rmtree('events')
        sys.exit()
###################################################################################################
###################################################################################################
