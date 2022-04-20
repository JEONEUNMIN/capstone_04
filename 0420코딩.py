#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import zipfile
# 기본 경로
base_dir = "C:\\Users\\akwld\\Desktop\\AI_Model"

train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# 훈련에 사용되는 이미지 경로
train_bicycle_dir = os.path.join(train_dir, 'bicycle')
train_side_dir = os.path.join(train_dir, 'side')
train_walking_dir = os.path.join(train_dir, 'walking')
train_road_dir = os.path.join(train_dir, 'road')

print(train_bicycle_dir)
print(train_side_dir)
print(train_walking_dir)
print(train_road_dir)

# 테스트에 사용되는 이미지 경로
test_bicycle_dir = os.path.join(test_dir, 'bicycle')
test_side_dir = os.path.join(test_dir, 'side')
test_walking_dir = os.path.join(test_dir, 'walking')
test_road_dir = os.path.join(test_dir, 'road')

print(test_bicycle_dir)
print(test_side_dir)
print(test_walking_dir)
print(test_road_dir)


# In[4]:


train_bicycle_fnames = os.listdir( train_bicycle_dir )
train_side_fnames = os.listdir( train_side_dir )
train_walking_fnames = os.listdir( train_walking_dir )
train_road_fnames = os.listdir( train_road_dir )

print(train_bicycle_fnames[:5])
print(train_side_fnames[:5])
print(train_walking_fnames[:5])
print(train_road_fnames[:5])


# In[5]:


print('Total training bicycle images :', len(os.listdir(train_bicycle_dir)))
print('Total training side images :', len(os.listdir(train_side_dir)))
print('Total training walking images :', len(os.listdir(train_walking_dir)))
print('Total training road images :', len(os.listdir(train_road_dir)))

print('Total test bicycle images :', len(os.listdir(test_bicycle_dir)))
print('Total test side images :', len(os.listdir(test_side_dir)))
print('Total test walking images :', len(os.listdir(test_walking_dir)))
print('Total test road images :', len(os.listdir(test_road_dir)))


# In[78]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator( rescale = 1.0/255. ) 
test_datagen  = ImageDataGenerator( rescale = 1.0/255. )

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    classes=['bicycle', 'side', 'walking', 'road'],
                                                  batch_size=10,
                                                  target_size=(96, 96),
                                                   class_mode='categorical')
test_generator =  test_datagen.flow_from_directory(test_dir,
                                                   classes=['bicycle', 'side', 'walking', 'road'],
                                                       batch_size=10,
                                                       target_size = (96, 96),
                                                  class_mode  = 'categorical')


# In[79]:


print(train_generator.class_indices) 
print(test_generator.class_indices)


# In[90]:


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization


model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same', input_shape=(96, 96, 3)),
  tf.keras.layers.MaxPooling2D(2,2),

  tf.keras.layers.Conv2D(64, (3,3), padding='same',activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
    
  tf.keras.layers.Conv2D(128, (3,3), padding='same',activation='relu'), 
  tf.keras.layers.MaxPooling2D(2,2), 
    
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.8),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dropout(0.8),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dropout(0.5), 
  tf.keras.layers.Dense(4, activation='softmax')
])

model.summary()


# In[93]:


model.compile(loss="categorical_crossentropy",optimizer="adam",  metrics = ['accuracy'])


# In[94]:


history = model.fit(train_generator, epochs=20, validation_data=test_generator)


# In[ ]:


import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'go', label='Training Loss')
plt.plot(epochs, val_loss, 'g', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# In[40]:


import numpy as np
#from google.colab import files
from keras.preprocessing import image
from keras.models import Sequential

#uploaded=files.upload()

#for fn in uploaded.keys():

path="C:\\Users\\akwld\\Desktop\\AI_Model\\side1.png"
img=image.load_img(path, target_size=(96, 96))
plt.imshow(img)

x=image.img_to_array(img)
x=np.expand_dims(x, axis=0)
images = np.vstack([x])

classes = model.predict(images, batch_size=10)

print(classes)

if classes[0][0]==1.0:
    print("is a bicycle")
elif classes[0][1]==1.0:
    print("is a side")
elif classes[0][2]==1.0:
    print("is a walking")
else:
    print("is a road")


# In[ ]:





# In[ ]:




