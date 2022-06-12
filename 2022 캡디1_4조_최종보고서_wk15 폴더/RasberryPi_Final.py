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

model = tf.keras.models.load_model(os.path.join("/home/capstone04/Desktop/h5files/","0530_test.h5"))

camera = picamera.PiCamera()
camera.resolution = (1920, 1080)

# 테스트용 코드, 10번만 실행되도록 설정
# while True:
for n in range(0,10):
    
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
    
    img = image.load_img(path, target_size=(128, 256))
    plt.imshow(img)
    
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    
    classes = model.predict(images)
    
    index = np.argmax(classes[0])
    
    if index==0:
        print("is a bicycle.")
        print("가능")
        pwm.stop()
        GPIO.cleanup()
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
        pwm.stop()
        GPIO.cleanup()
    elif index==3:
        print("is a walking")
        print("불가능")
        pwm.start(50.0)
        time.sleep(0.5)
        pwm.stop()
        GPIO.cleanup()
    
    time.sleep(3)
camera.stop_preview()
