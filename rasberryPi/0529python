import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from tensorflow import keras

import RPi.GPIO as GPIO
import picamera
import time
import datetime

from keras.models import load_model
import matplotlib.pyplot as plt
import os



model = tf.keras.models.load_model(os.path.join("/home/capstone04/Desktop/0521/","0526_model.h5"))

camera = picamera.PiCamera()
camera.resolution = (1024,768)

#while True:
for n in range(0,3):
    
    buzzer = 18
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(buzzer,GPIO.OUT)
    GPIO.setwarnings(False)

    pwm=GPIO.PWM(buzzer,262)
    now=datetime.datetime.now()
    filename = now.strftime('%Y-%m-%d %H:%M:%S')
    
    camera.start_preview()
    camera.capture('/home/capstone04/Image/'+filename+'.png')
    
    path = '/home/capstone04/Image/'+filename+'.png'
    
    img = image.load_img(path, target_size=(224, 224))
    plt.imshow(img)
    
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    
    classes = model.predict(images)
    # print(classes)
    
    index = np.argmax(classes[0])
    # print(index)
    
    if index==0:
        print("is a bicycle")
        print("가능")
    elif index==1:
        print("is a road")
        print("불가능")
        pwm.start(50.0)
        time.sleep(0.5)
        pwm.stop()
        GPIO.cleanup()
    elif index==2:
        print("is a side")
        print("가능")
    elif index==3:
        print("is a walking")
        print("불가능")
        pwm.start(50.0)
        time.sleep(0.5)
        pwm.stop()
        GPIO.cleanup()
    #print("성공")
    
    time.sleep(3)
camera.stop_preview()
