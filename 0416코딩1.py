#!/usr/bin/env python
# coding: utf-8

# In[192]:


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
print(train_bicycle_dir)
print(train_side_dir)
print(train_walking_dir)

# 테스트에 사용되는 이미지 경로
test_bicycle_dir = os.path.join(test_dir, 'bicycle')
test_side_dir = os.path.join(test_dir, 'side')
test_walking_dir = os.path.join(test_dir, 'walking')

print(test_bicycle_dir)
print(test_side_dir)
print(test_walking_dir)


# In[193]:


train_bicycle_fnames = os.listdir( train_bicycle_dir )
train_side_fnames = os.listdir( train_side_dir )
train_walking_fnames = os.listdir( train_walking_dir )

print(train_bicycle_fnames[:5])
print(train_side_fnames[:5])
print(train_walking_fnames[:5])


# In[194]:


print('Total training bicycle images :', len(os.listdir(train_bicycle_dir)))
print('Total training side images :', len(os.listdir(train_side_dir)))
print('Total training walking images :', len(os.listdir(train_walking_dir)))

print('Total test bicycle images :', len(os.listdir(test_bicycle_dir)))
print('Total test side images :', len(os.listdir(test_side_dir)))
print('Total test walking images :', len(os.listdir(test_walking_dir)))


# In[195]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator( rescale = 1.0/255. )
test_datagen  = ImageDataGenerator( rescale = 1.0/255. )

train_generator = train_datagen.flow_from_directory(train_dir,
                                                  batch_size=20,
                                                  class_mode='categorical',
                                                  target_size=(150, 150))
test_generator =  test_datagen.flow_from_directory(test_dir,
                                                       batch_size=20,
                                                       class_mode  = 'categorical',
                                                       target_size = (150, 150))


# In[196]:


print(train_generator.class_indices) 
print(test_generator.class_indices)


# In[197]:


import tensorflow as tf



model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3), padding='same'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3), padding='same'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(128, (3,3), activation='relu', input_shape=(150, 150, 3), padding='same'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(256, (3,3), activation='relu', input_shape=(150, 150, 3), padding='same'),
  tf.keras.layers.MaxPooling2D(2,2),
    
  tf.keras.layers.Flatten(),
    
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(3, activation='softmax')
])

model.summary()


# In[198]:


model.compile(loss="categorical_crossentropy",optimizer="adam",  metrics = ['accuracy'])


# In[199]:


history = model.fit(train_generator, epochs=30, validation_data=test_generator)


# In[200]:


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


# In[210]:


import numpy as np
#from google.colab import files
from keras.preprocessing import image
from keras.models import Sequential

#uploaded=files.upload()

#for fn in uploaded.keys():

path="C:\\Users\\akwld\\Desktop\\AI_Model\\test1.png"
img=image.load_img(path, target_size=(150, 150))
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
else:
    print("is a walking")

