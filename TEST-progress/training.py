import os
import matplotlib
import matplotlib.pylab as plt
import matplotlib.image as mpimg
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
from keras.preprocessing import image


# =================================================================================================== #
# 기본 경로
base_dir = "D:/New_Dataset"

train_dir = os.path.join(base_dir, 'train') #학습데이터 폴더 경로
validation_dir = os.path.join(base_dir, 'validation') #검증데이터 폴더 경로

# 학습에 사용되는 이미지 경로
train_bicycle_dir = os.path.join(train_dir, 'bicycle') 
train_bicycle_night_dir = os.path.join(train_dir, 'bicycle_night') 
train_road_dir = os.path.join(train_dir, 'road')   
train_side_dir = os.path.join(train_dir, 'side') 
train_side_night_dir = os.path.join(train_dir, 'side_night')       
train_walking_dir = os.path.join(train_dir, 'walking') 
train_walking_night_dir = os.path.join(train_dir, 'walking_night') 


# 검증에 사용되는 이미지 경로
validation_bicycle_dir = os.path.join(validation_dir, 'bicycle')
validation_bicycle_night_dir = os.path.join(validation_dir, 'bicycle_night')
validation_road_dir = os.path.join(validation_dir, 'road')
validation_side_dir = os.path.join(validation_dir, 'side')
validation_side_night_dir = os.path.join(validation_dir, 'side_night')
validation_walking_dir = os.path.join(validation_dir, 'walking')
validation_walking_night_dir = os.path.join(validation_dir, 'walking_night')

# =================================================================================================== #

#학습데이터 경로에 따른 디렉토리 목록을 불러옴
train_bicycle_fnames = os.listdir( train_bicycle_dir )
train_bicycle_night_fnames = os.listdir( train_bicycle_night_dir )
train_road_fnames = os.listdir( train_road_dir )
train_side_fnames = os.listdir( train_side_dir )
train_side_night_fnames = os.listdir( train_side_night_dir )
train_walking_fnames = os.listdir( train_walking_dir )
train_walking_night_fnames = os.listdir( train_walking_night_dir )

#각 목록에 존재하는 이미지 파일 5개 불러와서 확인
print(train_bicycle_fnames[:5])
print(train_bicycle_night_fnames[:5])
print(train_road_fnames[:5])
print(train_side_fnames[:5])
print(train_side_night_fnames[:5])
print(train_walking_fnames[:5])
print(train_walking_night_fnames[:5])

# =================================================================================================== #

#학습 디렉토리 안에 이미지 갯수 출력
print('Total training bicycle images :', len(os.listdir(train_bicycle_dir)))
print('Total training bicycle_night images :', len(os.listdir(train_bicycle_night_dir)))
print('Total training road images :', len(os.listdir(train_road_dir)))
print('Total training side images :', len(os.listdir(train_side_dir)))
print('Total training side_night images :', len(os.listdir(train_side_night_dir)))
print('Total training walking images :', len(os.listdir(train_walking_dir)))
print('Total training walking_night images :', len(os.listdir(train_walking_night_dir)))

#검증 디렉토리 안에 이미지 갯수 출력
print('Total validation bicycle images :', len(os.listdir(validation_bicycle_dir)))
print('Total validation bicycle_night images :', len(os.listdir(validation_bicycle_night_dir)))
print('Total validation road images :', len(os.listdir(validation_road_dir)))
print('Total validation side images :', len(os.listdir(validation_side_dir)))
print('Total validation side_night images :', len(os.listdir(validation_side_night_dir)))
print('Total validation walking images :', len(os.listdir(validation_walking_dir)))
print('Total validation walking_night images :', len(os.listdir(validation_walking_night_dir)))

# =================================================================================================== #

train_datagen = ImageDataGenerator( rescale = 1.0/255. ) 
validation_datagen  = ImageDataGenerator( rescale = 1.0/255. )

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    classes=['bicycle', 'bicycle_night', 'road', 'side', 'side_night', 'walking', 'walking_night' ], 
                                                    batch_size=10,
                                                    target_size=(96, 96), 
                                                    class_mode='categorical')
validation_generator =  validation_datagen.flow_from_directory(validation_dir,
                                                    classes=['bicycle', 'bicycle_night', 'road', 'side', 'side_night', 'walking', 'walking_night' ],
                                                    batch_size=10,
                                                    target_size = (96, 96),
                                                    class_mode  = 'categorical')

# =================================================================================================== #

print(train_generator.class_indices) 
print(validation_generator.class_indices)

# =================================================================================================== #

model = tf.keras.models.Sequential([   
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(96, 96, 3)),
    tf.keras.layers.Conv2D(32, (3,3), padding='same',activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), padding='same',activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
        
    tf.keras.layers.Conv2D(128, (3,3), padding='same',activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2), 

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.8),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.8),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.5), 
    tf.keras.layers.Dense(7, activation='softmax')
])
model.summary()  

# =================================================================================================== #

model.compile(loss="categorical_crossentropy",optimizer="adam",  metrics = ['accuracy'])

# =================================================================================================== #

history = model.fit(train_generator, batch_size=256, epochs=20, validation_data=validation_generator)

model.save("modeltest.h5")

# =================================================================================================== #

hist_df = pd.DataFrame(history.history) 

hist_csv_file = 'history.csv'

with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

# =================================================================================================== #

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

# =================================================================================================== #

path = "side_test.png"
img=image.load_img(path, target_size=(96, 96))
plt.imshow(img)

x=image.img_to_array(img)
x=np.expand_dims(x, axis=0)
images = np.vstack([x])

classes = model.predict(images)

index = np.argmax(classes[0])

print(index)

if index==0 or index==1:   
    print("is a bicycle")
elif index==2:  
    print("is a road")
elif index==3 or index==4:  
    print("is a side")
elif index==5 or index==6:  
    print("is a walking")