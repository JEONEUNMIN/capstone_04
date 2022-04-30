
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





























#loss: 손실함수, 모델 최적화에 사용되는 목적 함수, 3개 이상의 클래스를 분류하는 multiclassification이라서 "categorical_crossentropy"을 사용해야함
#다중 분류 손실함수로 출력값이 one-hot encoding 된 결과로 나오고 실측 결과와의 비교시에도 실측 결과는 one-hot encoding 형태로 구성
#네트웍 레이어 구성시 마지막에 Dense(3, activation='softmax') 로 3개의 클래스 각각 별로 positive 확률값이 나옴

#optimizer: 최적화 알고리즘을 설정
#   Adam: RMSProp와 Momentum 기법을 합친 optimizer, Adagrad, Adadelta, RMSprop 처럼 각 파라미터마다 다른 크기의 업데이트를 적용하는 방법

#metrics: 훈련과정을 모니터링하는 방식을 지정, accuracy를 지정하면 학습과정에서 정확도를 수집
model.compile(loss="categorical_crossentropy", optimizer="adam",  metrics = ['accuracy'])





























history = model.fit(train_generator, #학습할 입력데이터
                    batch_size=256, #batch_size: 몇 개의 샘플로 가중치를 갱신할 것인지 지정(20이면 20개를 학습하고 가중치를 갱신, 전체 데이터가 100이라면 가중치 갱신은 100/20=5번 일어남) 
                    #batch_size를 크게하면 1epochs 시간이 줄어듬, size를 크게하고 epoch를 많이 할 것인지, 작게하고 epoch를 덜 할 것인지 결정해야함
                    epochs=30, #학습 반복 횟수
                    validation_data=validation_generator #검증 데이터
                   )
#model.fit() 함수의 리턴 값인 학습이력(history)을 history에 저장
#매 epoch마다 loss(훈련 손실값), accuracy(훈련 정확도), val_loss(검증 손실값), val_acc(검증 정확도) 값들이 저장되어있음




























#임의의 입력에 대한 모델의 출력값을 확인
classes = model.predict(images)           # 예측하고 싶은 데이터(테스트 이미지데이터)


index=np.argmax(classes[0]) #분류된 결과값 중에 가장 큰 값의 인덱스

if index==0:   
    print("is a bicycle")
elif index==1:
    print("is a road")
elif index==2:
    print("is a side")
else:
    print("is a walking")
