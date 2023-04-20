import uvicorn
import io
from fastapi import FastAPI
import numpy as np
import pandas as pd
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import save_img
import tensorflow as tf
from numpy import expand_dims, uint8
import cv2
import base64
from PIL import Image
from type import GAN
import os
from itertools import product
import tensorflow as tf
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

app = FastAPI()

def load_image(filename, size=(256,256)):
    pixels = load_img(filename, target_size=size)
    pixels = img_to_array(pixels)
    pixels = (pixels - 127.5) / 127.5
    pixels = expand_dims(pixels, 0)
    return pixels
def load_img2(filename):
    pixels = load_img(filename)
    pixels = img_to_array(pixels)
    pixels = (pixels - 127.5) / 127.5
    pixels = expand_dims(pixels, 0)
    
@app.get('/')
def home():
    return 'hello'
m1 = load_model('model.h5')
m2= load_model('model_.h5')
@app.post('/generate')
def generate(data):
    data=data.dict()
    if(data['type']==0):
        model=m1
        print('loaded model')
    else:
        model = m2
        print('loaded model')
    image = data['image']
    image = base64.b64decode(image)    
    preds=model.predict(image)
    output = tf.reshape(preds,[256,256,3])
    output = (output+1)/2.0
    print(output)
    outputimg = 'xyz.png'
    save_img(outputimg,img_to_array(output))
    with open(outputimg, "rb") as img_file:
        my_string = base64.b64encode(img_file.read())
    print(my_string)
    response ={
        "image" : my_string
    }
    return response
m1= load_model('model.h5')
m2 = load_model('model_.h5')
@app.post('/hi')
def hi(data:GAN):
    data=data.dict()
    if(data['type']==0):
        model=m1
        print('loaded model')
    else:
        model=m2
        print('loaded model')
    image = data['image']
    print(image)
    image = base64.b64decode(image+str(b'=='))
    with open("imageToSave.jpg", "wb") as fh:
        fh.write(image)
    def tile(filename, dir_in, dir_out, d):
        name, ext = os.path.splitext(filename)
        img = Image.open(os.path.join(dir_in, filename))
        w, h = img.size
        
        grid = product(range(0, h-h%d, d), range(0, w-w%d, d))
        for i, j in grid:
            box = (j, i, j+d, i+d)
            out = os.path.join(dir_out, f'{name}_{i//256}_{j//256}{ext}')
            img.crop(box).save(out)
    tile('imageToSave.jpg','','./output/',256)
    for filename in os.listdir('./output/'):
        img1 = load_image(os.path.join('./output/',filename))
        gen_image = model.predict(img1)
        gen_image = (gen_image + 1) / 2.0
        output=gen_image[0]
        img = img_to_array(output)
        my_string = base64.b64encode(img)
        img = Image.fromarray((gen_image[0]*255).astype(uint8))
        img.save(os.path.join('./pred/',filename))
    l=[]
    for i in range(0,29):
        r=[]
        for j in range(0,23):
            p = f"./pred/ImageToSave_{i}_{j}.jpg"
            r.append(cv2.imread(p))
        print(l)
        l.append(cv2.hconcat(r))
    result = cv2.vconcat(l)
    cv2.imwrite('xyz.png', result)
    with open("xyz.png", "rb") as img_file:
        my_string = base64.b64encode(img_file.read())
    print(image)
    
    response ={
        "image" : "image"
    }

# if __name__ == '__main__':
#     uvicorn.run(app, host='127.0.0.1', port=5000)