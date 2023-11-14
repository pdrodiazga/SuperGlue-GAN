"""
# > Script for testing FUnIE-GAN 
# > Notes and Usage:
#    - set data_dir and model paths
#    - python test_funieGAN.py
"""
import os
import cv2
import time
import ntpath
import numpy as np
from PIL import Image
from os.path import join, exists
from keras.models import model_from_json
## local libs
from utils.data_utils import getPaths, read_and_resize, preprocess, deprocess

## for testing arbitrary local data
# data_dir = "../data/test/A/"
# data_dir = "data"
from utils.data_utils import get_local_test_data
# test_paths = [r"C:\Users\Rahul\Downloads\archive\train_val\images\w_r_236_.jpg"]
img_path = [r"C:\Users\pdrod\Desktop\SuperGlue\FUnieGan-main\vid"]


## create dir for log and (sampled) validation data
samples_dir = "data"
if not exists(samples_dir): os.makedirs(samples_dir)

## test funie-gan
checkpoint_dir  = 'models/gen_p/'
model_name_by_epoch = "model_15320_" 
## test funie-gan-up
#checkpoint_dir  = 'models/gen_up/'
#model_name_by_epoch = "model_35442_" 

model_h5 = checkpoint_dir + model_name_by_epoch + ".h5"  
model_json = checkpoint_dir + model_name_by_epoch + ".json"
# sanity
assert (exists(model_h5) and exists(model_json))

# load model
with open(model_json, "r") as json_file:
    loaded_model_json = json_file.read()
funie_gan_generator = model_from_json(loaded_model_json)
# load weights into new model
funie_gan_generator.load_weights(model_h5)
print("\nLoaded data and model")
cap = cv2.VideoCapture('GOPR3749.mp4')
output=cv2.VideoWriter_fourcc(*'mp4v')
output_file='FunieGan.mp4'
# Especifica el tamaño del frame (ancho, alto) y la velocidad de fotogramas (FPS)
frame_width = 640
frame_height = 352
fps = 60
out = cv2.VideoWriter(output_file, output, fps, (frame_width, frame_height))
print("Loaded video")
# testing loop
times = []; s = time.time()
while True:
    # prepare data
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (256,256))
    im = preprocess(frame)
    im = np.expand_dims(im, axis=0) # (1,256,256,3)
    # generate enhanced image
    s = time.time()
    # im.shape[3] = 3
    # new_im = (im.shape[0],im.shape[1],im.shape[2],3)
    gen = funie_gan_generator.predict(im)
    gen_img = deprocess(gen)[0]
    tot = time.time()-s
    times.append(tot)
    out_img = np.hstack((frame, gen_img)).astype('uint8')
    # Image.fromarray(gen_img).save(join(samples_dir, img_name))
    frame = cv2.resize(gen_img, (frame_width, frame_height))
    out.write(frame)
    cv2.imshow('Enchanced frame', gen_img)
    cv2.imshow('Comparison', out_img)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        print('stop_dispose')
        break


