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

# 기본 경로
base_dir = 'D:\Clear_Dataset'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# 훈련에 사용되는 자전거도로, 일반도로, 갓길, 인도 이미지 경로
train_bicycle_dir = os.path.join(train_dir, 'bicycle')
train_road_dir = os.path.join(train_dir, 'road')
train_side_dir = os.path.join(train_dir, 'side')
train_walking_dir = os.path.join(train_dir, 'walking')
# 테스트에 사용되는 자전거도로, 일반도로, 갓길, 인도 이미지 경로
validation_bicycle_dir = os.path.join(validation_dir, 'bicycle')
validation_road_dir = os.path.join(validation_dir, 'road')
validation_side_dir = os.path.join(validation_dir, 'side')
validation_walking_dir = os.path.join(validation_dir, 'walking')


train_bicycle_fnames = os.listdir( train_bicycle_dir )
train_road_fnames = os.listdir( train_road_dir )
train_side_fnames = os.listdir( train_side_dir )
train_walking_fnames = os.listdir( train_walking_dir )


print('Total training bicycle images :', len(os.listdir(train_bicycle_dir)))
print('Total training road images :', len(os.listdir(train_road_dir)))
print('Total training side images :', len(os.listdir(train_side_dir)))
print('Total training walking images :', len(os.listdir(train_walking_dir)))

print('Total validation bicycle images :', len(os.listdir(validation_bicycle_dir)))
print('Total validation road images :', len(os.listdir(validation_road_dir)))
print('Total validation side images :', len(os.listdir(validation_side_dir)))
print('Total validation walking images :', len(os.listdir(validation_walking_dir)))

train_datagen = ImageDataGenerator( rescale = 1.0/255. ) 
validation_datagen  = ImageDataGenerator( rescale = 1.0/255. )

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    classes=['bicycle', 'road', 'side', 'walking' ],
                                                    batch_size=10,
                                                    target_size=(128, 128),
                                                    class_mode='categorical')
validation_generator =  validation_datagen.flow_from_directory(validation_dir,
                                                    classes=['bicycle', 'road', 'side', 'walking' ],
                                                    batch_size=10,
                                                    target_size = (128, 128),
                                                    class_mode  = 'categorical')


print(train_generator.class_indices) 
print(validation_generator.class_indices)


model = tf.keras.models.Sequential([    #각 레이어에 정확히 하나의 입력 텐서와 하나의 출력 텐서가 있는 레이어 스택에 적합한 모델
    # padding : 경계처리, 데이터의 손실을 방지하기 위해 same으로 설정
    # input_shape : 샘플 수를 제외한 입력 형태를 정의함
    # activation : 활성화 함수, 은닉층에 자주 사용되는 (if x > 0 면 x, x <= 0 면 0)인 relu 사용
    # 첫 번째 인자 : 필터의 개수, 두 번째 인자 : 필터의 크기
    # conv2d : 합성곱 계층, 이미지의 특징을 추출하여 패턴파악에 사용
    # tf.keras.layers.Conv2D(32, (5,5), activation='relu', padding='same', input_shape=(128, 128, 3)),
    # tf.keras.layers.Conv2D(32, (5,5), padding='same',activation='relu'),
    # # pooling : 지역적 집계함수를 사용해 데이터의 크기를 줄이는 과정
    # tf.keras.layers.MaxPooling2D(2,2),

    # tf.keras.layers.Conv2D(64, (3,3), padding='same',activation='relu'),
    # tf.keras.layers.Conv2D(64, (3,3), padding='same',activation='relu'),
    # tf.keras.layers.MaxPooling2D(2,2),
        
    # tf.keras.layers.Conv2D(128, (3,3), padding='same',activation='relu'), 
    # tf.keras.layers.MaxPooling2D(2,2), 

    # # Flatten : conv2d -> pooling 거친 2차원 데이터를 평탄화시켜 1차원 데이터로 만들어 냄
    # tf.keras.layers.Flatten(),
    # # Dense : 은닉층, 뉴런의 입력과 출력을 연결해주는 역할
    # tf.keras.layers.Dense(128, activation='relu'),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dropout(0.8),

    # tf.keras.layers.Dense(64, activation='relu'),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dropout(0.8),
    
    # tf.keras.layers.Dense(16, activation='relu'),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dropout(0.5), 
    
    tf.keras.layers.Conv2D(200, (3,3), activation='relu', padding='same', input_shape=(128, 128, 3)),
    tf.keras.layers.Conv2D(150, (3,3), padding='same',activation='relu'),
    # pooling : 지역적 집계함수를 사용해 데이터의 크기를 줄이는 과정
    tf.keras.layers.MaxPooling2D(4,4),

    tf.keras.layers.Conv2D(120, (3,3), padding='same',activation='relu'),
    tf.keras.layers.Conv2D(80, (3,3), padding='same',activation='relu'),
    tf.keras.layers.Conv2D(50, (3,3), padding='same',activation='relu'), 
    tf.keras.layers.MaxPooling2D(4,4), 

    # Flatten : conv2d -> pooling 거친 2차원 데이터를 평탄화시켜 1차원 데이터로 만들어 냄
    tf.keras.layers.Flatten(),
    # Dense : 은닉층, 뉴런의 입력과 출력을 연결해주는 역할
    tf.keras.layers.Dense(120, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dropout(0.5), 
    

    tf.keras.layers.Dense(4, activation='softmax')
])
model.summary()     # 모델 구조를 확인

model.compile(loss="categorical_crossentropy",optimizer="adam",  metrics = ['accuracy'])

history = model.fit(train_generator, batch_size=96, epochs=20, validation_data=validation_generator)

model.save("modeltest.h5")

hist_df = pd.DataFrame(history.history) 

hist_csv_file = 'history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

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


path="C:\\Users\\이수정\\Desktop\\capstone\\bicycle_img.jpg"
img=image.load_img(path, target_size=(128,128))
plt.imshow(img)

x=image.img_to_array(img)
x=np.expand_dims(x, axis=0)
images = np.vstack([x])

classes = model.predict(images, batch_size=10)

maxRes = np.argmax(classes[0])


print(maxRes)

if maxRes == 0:
    print("is a bicycle.")
elif maxRes == 1:
    print("is a road")
elif maxRes == 2:
    print("is a side")
elif maxRes == 3:
    print("is a walking")