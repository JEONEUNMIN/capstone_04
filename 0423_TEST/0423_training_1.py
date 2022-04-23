import os 
import matplotlib
import matplotlib.pylab as plt
import matplotlib.image as mpimg

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
from keras.preprocessing import image

# 기본 경로
base_dir = '..\dataset'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# 훈련에 사용되는 자전거도로, 일반도로, 갓길, 인도 이미지 경로
train_bicycle_dir = os.path.join(train_dir, 'bicycle')
train_road_dir = os.path.join(train_dir, 'road')
train_side_dir = os.path.join(train_dir, 'side')
train_walking_dir = os.path.join(train_dir, 'walking')
# print(train_bicycle_dir)
# print(train_road_dir)
# print(train_side_dir)
# print(train_walking_dir)

# 테스트에 사용되는 자전거도로, 일반도로, 갓길, 인도 이미지 경로
validation_bicycle_dir = os.path.join(validation_dir, 'bicycle')
validation_road_dir = os.path.join(validation_dir, 'road')
validation_side_dir = os.path.join(validation_dir, 'side')
validation_walking_dir = os.path.join(validation_dir, 'walking')
# print(validation_bicycle_dir)
# print(validation_road_dir)
# print(validation_side_dir)
# print(validation_walking_dir)


train_bicycle_fnames = os.listdir( train_bicycle_dir )
train_road_fnames = os.listdir( train_road_dir )
train_side_fnames = os.listdir( train_side_dir )
train_walking_fnames = os.listdir( train_walking_dir )

# print(train_bicycle_fnames[:5])
# print(train_road_fnames[:5])
# print(train_side_fnames[:5])
# print(train_walking_fnames[:5])


# print('Total training bicycle images :', len(os.listdir(train_bicycle_dir)))
# print('Total training road images :', len(os.listdir(train_road_dir)))
# print('Total training side images :', len(os.listdir(train_side_dir)))
# print('Total training walking images :', len(os.listdir(train_walking_dir)))

# print('Total validation bicycle images :', len(os.listdir(validation_bicycle_dir)))
# print('Total validation road images :', len(os.listdir(validation_road_dir)))
# print('Total validation side images :', len(os.listdir(validation_side_dir)))
# print('Total validation walking images :', len(os.listdir(validation_walking_dir)))


# nrows, ncols = 8, 4
# pic_index = 0

# fig = plt.gcf()
# fig.set_size_inches(ncols*3, nrows*3)

# pic_index+=8

# next_bicycle_pix = [os.path.join(train_bicycle_dir, fname)
#                 for fname in train_bicycle_fnames[ pic_index-8:pic_index]]

# next_road_pix = [os.path.join(train_road_dir, fname)
#                 for fname in train_road_fnames[ pic_index-8:pic_index]]

# next_side_pix = [os.path.join(train_side_dir, fname)
#                 for fname in train_side_fnames[ pic_index-8:pic_index]]

# next_walking_pix = [os.path.join(train_walking_dir, fname)
#                 for fname in train_walking_fnames[ pic_index-8:pic_index]]

# for i, img_path in enumerate(next_bicycle_pix+next_side_pix+next_walking_pix):
#   sp = plt.subplot(nrows, ncols, i + 1)
#   sp.axis('Off')

#   img = mpimg.imread(img_path)
#   plt.imshow(img)
# plt.show()


train_datagen = ImageDataGenerator( rescale = 1.0/255. ) 
validation_datagen  = ImageDataGenerator( rescale = 1.0/255. )

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    classes=['bicycle','road', 'side', 'walking' ],
                                                  batch_size=10,
                                                  target_size=(96, 96),
                                                   class_mode='categorical')
validation_generator =  validation_datagen.flow_from_directory(validation_dir,
                                                   classes=['bicycle','road', 'side', 'walking' ],
                                                       batch_size=10,
                                                       target_size = (96, 96),
                                                  class_mode  = 'categorical')


print(train_generator.class_indices) 
print(validation_generator.class_indices)


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


model.compile(loss="categorical_crossentropy",optimizer="adam",  metrics = ['accuracy'])

history = model.fit(train_generator, epochs=20, validation_data=validation_generator)



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


path="C:\\Users\\이수정\\Desktop\\side_test.png"
img=image.load_img(path, target_size=(96, 96))
plt.imshow(img)

x=image.img_to_array(img)
x=np.expand_dims(x, axis=0)
images = np.vstack([x])

# classes = model.predict(images, batch_size=10)

y_prob = model.predict(images, verbose=0)
classes = y_prob.argmax(axis=-1)


print(classes)

if classes[0][0]==1.0:
    print("is a bicycle")
elif classes[0][1]==1.0:
    print("is a side")
elif classes[0][2]==1.0:
    print("is a walking")
else:
    print("is a road")