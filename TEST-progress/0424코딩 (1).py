#!/usr/bin/env python
# coding: utf-8

# In[127]:


#운영체제에서 제공되는 여러 기능을 파이썬에서 수행할 수 있게함
#특정 경로에 존재하는 파일과 디렉터리 목록을 구하는 함수인 listdir 함수를 사용하기 위해 사용
import os

# 기본 경로: 데이터가 들어있는 폴더 경로
base_dir = "C:\\Users\\akwld\\Desktop\\AI_Model"

train_dir = os.path.join(base_dir, 'train') #학습데이터 폴더 경로
validation_dir = os.path.join(base_dir, 'validation') #검증데이터 폴더 경로

# 학습에 사용되는 이미지 경로
train_bicycle_dir = os.path.join(train_dir, 'bicycle') #label: bicycle
train_road_dir = os.path.join(train_dir, 'road')       #label: road
train_side_dir = os.path.join(train_dir, 'side')       #lable: side
train_walking_dir = os.path.join(train_dir, 'walking') #label: walking

# 검증에 사용되는 이미지 경로
validation_bicycle_dir = os.path.join(validation_dir, 'bicycle')
validation_road_dir = os.path.join(validation_dir, 'road')
validation_side_dir = os.path.join(validation_dir, 'side')
validation_walking_dir = os.path.join(validation_dir, 'walking')


# In[128]:


#학습데이터 경로에 따른 디렉토리 목록을 불러옴
train_bicycle_fnames = os.listdir( train_bicycle_dir )  
train_road_fnames = os.listdir( train_road_dir )
train_side_fnames = os.listdir( train_side_dir )
train_walking_fnames = os.listdir( train_walking_dir )

#각 목록에 존재하는 이미지 파일 5개 불러와서 확인
print(train_bicycle_fnames[:5])
print(train_side_fnames[:5])
print(train_walking_fnames[:5])
print(train_road_fnames[:5])


# In[130]:


#학습 디렉토리 안에 이미지 갯수 출력
print('Total training bicycle images :', len(os.listdir(train_bicycle_dir)))
print('Total training road images :', len(os.listdir(train_road_dir)))
print('Total training side images :', len(os.listdir(train_side_dir)))
print('Total training walking images :', len(os.listdir(train_walking_dir)))

#검증 디렉토리 안에 이미지 갯수 출력
print('Total validation bicycle images :', len(os.listdir(validation_bicycle_dir)))
print('Total validation road images :', len(os.listdir(validation_road_dir)))
print('Total validation side images :', len(os.listdir(validation_side_dir)))
print('Total validation walking images :', len(os.listdir(validation_walking_dir)))


# In[142]:


#객체를 생성할 때 파라미터를 전달해주는 것을 통해 데이터 전처리
#rescale: 원본 영상은 0-255의 RGB 계수로 구성, 이런 입력값은 모델 학습시키기에 너무 높음, 그래서 1.0/255로 스케일링하여 0-1 범위로 변환
train_datagen = ImageDataGenerator( rescale = 1.0/255. ) 
validation_datagen  = ImageDataGenerator( rescale = 1.0/255. )

#flow_from_directory 메소드를 사용해서 데이터 폴더구조를 그대로 가져와서 ImageDataGenerator 객체(train_generator, validation_datagen)에 실제 데이터를 채움
train_generator = train_datagen.flow_from_directory(train_dir, #표적 디렉토리 경로(한 클래스 당 하나의 하위 디렉토리가 있어야함 )
                                                    classes=['bicycle', 'road', 'side', 'walking' ], #클래스 하위 디렉토리의 선택적 리스트(하위 디렉토리의 이름/구조에서 자동으로 유추됨), 4개의 클래스로 구분
                                                    batch_size=10, #폴더 안의 이미지 한번에 읽어들일 양
                                                    target_size=(96, 96), #사용할 CNN모델 입력 사이즈에 맞게 조정, 원본 이미지 크기가 다르더라도 지정된 크기로 자동조정
                                                    class_mode='categorical') #categorical: 멀티-레이블 클래스, one-hot 부호화된 라벨이 반환
validation_generator =  validation_datagen.flow_from_directory(validation_dir,
                                                    classes=['bicycle', 'road', 'side', 'walking' ],
                                                    batch_size=10,
                                                    target_size = (96, 96),
                                                    class_mode  = 'categorical')

#전체 학습 데이터와 검증 데이터의 양이 출력됨


# In[138]:


#class_indices 속성을 통해서 클래스 이름과 클래스 색인 간 매핑을 담은 딕셔너리 출력
print(train_generator.class_indices) 
print(validation_generator.class_indices)


# In[147]:


import tensorflow as tf     #모델 구성을 위해 라이브러리 임포드

#모델 구성을 위한 라이브러리
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

model = tf.keras.models.Sequential([    #각 레이어에 정확히 하나의 입력 텐서와 하나의 출력 텐서가 있는 레이어 스택에 적합한 모델
    # padding : 경계처리, 데이터의 손실을 방지하기 위해 same으로 설정
    # input_shape : 샘플 수를 제외한 입력 형태를 정의함
    # activation : 활성화 함수, 은닉층에 자주 사용되는 (if x > 0 면 x, x <= 0 면 0)인 relu 사용
    # 첫 번째 인자 : 필터의 개수, 두 번째 인자 : 필터의 크기
    # conv2d : 합성곱 계층, 이미지의 특징을 추출하여 패턴파악에 사용
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(96, 96, 3)),
    tf.keras.layers.Conv2D(32, (3,3), padding='same',activation='relu'),
    # pooling : 지역적 집계함수를 사용해 데이터의 크기를 줄이는 과정
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), padding='same',activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
        
    tf.keras.layers.Conv2D(128, (3,3), padding='same',activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2), 

    # Flatten : conv2d -> pooling 거친 2차원 데이터를 평탄화시켜 1차원 데이터로 만들어 냄
    tf.keras.layers.Flatten(),
    # Dense : 은닉층, 뉴런의 입력과 출력을 연결해주는 역할
    tf.keras.layers.Dense(256, activation='relu'),
    # Dropout : Overfitting을 해소하기 위한 방법, 은닉층의 일부 유닛을 동작하지 않게 하여 과적합을 막음
    tf.keras.layers.Dropout(0.8),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.8),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.5), 
    # 첫번째 인자 : 출력 뉴런의 수
    # activation='softmax' : 활성화 함수 softmax는 다중클래스 분류의 마지막 출력층에서 사용
    tf.keras.layers.Dense(4, activation='softmax')
])
model.summary()     # 모델 구조를 확인


# In[140]:


#loss: 손실함수, 모델 최적화에 사용되는 목적 함수, 3개 이상의 클래스를 분류하는 multiclassification이라서 "categorical_crossentropy"을 사용해야함
#다중 분류 손실함수로 출력값이 one-hot encoding 된 결과로 나오고 실측 결과와의 비교시에도 실측 결과는 one-hot encoding 형태로 구성
#네트웍 레이어 구성시 마지막에 Dense(3, activation='softmax') 로 3개의 클래스 각각 별로 positive 확률값이 나옴

#optimizer: 최적화 알고리즘을 설정
#   Adam: RMSProp와 Momentum 기법을 합친 optimizer, Adagrad, Adadelta, RMSprop 처럼 각 파라미터마다 다른 크기의 업데이트를 적용하는 방법

#metrics: 훈련과정을 모니터링하는 방식을 지정, accuracy를 지정하면 학습과정에서 정확도를 수집
model.compile(loss="categorical_crossentropy",optimizer="adam",  metrics = ['accuracy'])


# In[144]:


history = model.fit(train_generator, #학습할 입력데이터
                    batch_size=256, #batch_size: 몇 개의 샘플로 가중치를 갱신할 것인지 지정(20이면 20개를 학습하고 가중치를 갱신, 전체 데이터가 100이라면 가중치 갱신은 100/20=5번 일어남) 
                    #batch_size를 크게하면 1epochs 시간이 줄어듬, size를 크게하고 epoch를 많이 할 것인지, 작게하고 epoch를 덜 할 것인지 결정해야함
                    epochs=30, #학습 반복 횟수
                    validation_data=validation_generator #검증 데이터
                   )
#model.fit() 함수의 리턴 값인 학습이력(history)을 history에 저장
#매 epoch마다 loss(훈련 손실값), accuracy(훈련 정확도), val_loss(검증 손실값), val_acc(검증 정확도) 값들이 저장되어있음


# In[145]:


#모델 학습 결과를 그림으로 표현하기
import matplotlib.pyplot as plt

acc = history.history['accuracy']           #history에 저장된 훈련 정확도
val_acc = history.history['val_accuracy']   #history에 저장된 검증 정확도
loss = history.history['loss']              #history에 저장된 훈련 손실값
val_loss = history.history['val_loss']      #history에 저장된 검증 손실값

epochs = range(len(acc)) #몇 epoch 돌렸는지=정확도 갯수

#훈련 정확도, 검증 정확도 그래프 2개가 동시에 그려짐
#훈련 정확도 그래프
plt.plot(epochs,    # x축 범위: epoch 수
         acc,       # y축 범위: 훈련 정확도 범위
         'bo',      # 'bo'로 지정하면 파란색 원형 마커로 그래프에 표시됨
         label='Training accuracy') 

#검증 정확도 그래프
plt.plot(epochs,    # x축 범위: epoch 수
         val_acc,   # y축 범위: 검증 정확도 범위
         'b',       # 'b'로 지정하면 파란색 선으로 그래프에 표시됨
         label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()  #범례 표시(어떤 그래프가 훈련 정확도를 나타내는지, 검증 정확도를 나타내는지 표시)

plt.figure() #기본값으로 설정된 새로운 figure 생성

plt.plot(epochs, loss, 'go', label='Training Loss') # 'go'로 지정하면 초록색(green) 원형 마커로 그래프에 표시됨
plt.plot(epochs, val_loss, 'g', label='Validation Loss') # 'g'로 지정하면 초록색(green) 선으로 그래프에 표시됨
plt.title('Training and validation loss')
plt.legend()  #범례 표시(어떤 그래프가 훈련 손실값를 나타내는지, 검증 손실값를 나타내는지 표시)


plt.show() #그래프 보이기


# In[150]:


import numpy as np
from keras.preprocessing import image
from keras.models import Sequential

path="C:\\Users\\akwld\\Desktop\\side.jpg" #test 할 이미지데이터의 경로
img=image.load_img(path, target_size=(96, 96)) #경로에서 이미지를 가져와서 크기를 96,96으로 조정하기
plt.imshow(img) #이미지 보이기

x=image.img_to_array(img) #테스트 이미지를 배열로 변환
x=np.expand_dims(x, axis=0) #이미지를 배열로 변환한 x를 받아서 numpy배열에 차원을 추가, 배열을 2D배열로 변환, axis=0: 첫번째 차원에 새로운 축을 만듬
images = np.vstack([x]) #배열 결합시키기, vstack: 배열을 세로로 결합

#임의의 입력에 대한 모델의 출력값을 확인
classes = model.predict(images,           # 예측하고 싶은 데이터(테스트 이미지데이터)
                        batch_size=10)    # 배치크기
print(classes) #분류된 클래스의 결과값 출력

index=np.argmax(classes[0]) #분류된 결과값 중에 가장 큰 값의 인덱스
print(index)

#클래스 라벨 순서: 'bicycle', 'road', 'side', 'walking'

if index==0:   
    print("is a bicycle")
elif index==1:
    print("is a road")
elif index==2:
    print("is a side")
else:
    print("is a walking")


# In[ ]:




