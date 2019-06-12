from cv2 import cv2
import tensorflow as tf
from keras.models import load_model
import os

CATEGORIES = ["Dog", "Cat"]

def prepare(filepath):
    IMG_SIZE = 50
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = load_model('64x3-CNN.model')
image_path = input ("Enter the name of an image + its format (example cat01.jpg): ")
prediction = model.predict([prepare(os.getcwd()+"\\Photos\\"+image_path)])
print("It's a " + CATEGORIES[int(prediction[0][0])] + "!")
os.system("pause")