import os
import csv
import time
import h5py
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.regularizers import l2
from keras.models import Model, load_model
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.losses import binary_crossentropy
import keras.backend.tensorflow_backend as KTF

SRC_PATH = '/home/ashish/Dev/CarND-Behavioral-Cloning-P3'
DATA_PATH = '/home/ashish/Dev/CarND-Behavioral-Cloning-P3-data'
RUNS = ['run11','run22','run23','run25','run30','run31','run32','run16','run1','run2','run3']
#['run1','run2','run3','run22','run23','run24','run25','run12','run13','run14','run15']
BATCH_SIZE = 8
TENSORBOARD_PATH = os.path.join(SRC_PATH, 'tensorboard')
MODELS_PATH = os.path.join(SRC_PATH, 'models')
IMG_SHAPE = (160, 320, 3)
STEERING_CORRECTION = 0.2
ADD_FLIPS = True
ADD_SIDE_VIEWS = True
KEEP_ZERO = False
KEEP_ZERO_STEERING_ANGLE_THRESHOLD = 0.4
NORMALIZE_BRIGHTNESS = False

headers = ['center_img_path', 'left_img_path', 'right_img_path', 'steering_angle', 'throttle', 'brake', 'speed']
lines = []
runs = ""

def normalize_brightness(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    # convert the YUV image back to RGB format
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

for idx, run in enumerate(RUNS):
    run_path = os.path.join(DATA_PATH, run)
    driving_log_path = os.path.join(run_path, 'driving_log.csv')

    runs = runs + RUNS[idx] + ","
    with open(driving_log_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

if not KEEP_ZERO:
    temp = []
    np.random.seed(999)
    # Skip 0 deg steering angle images
    for line in lines:
        if float(line[3].strip()) != 0.0:
            temp.append(line)
        elif np.random.uniform() > KEEP_ZERO_STEERING_ANGLE_THRESHOLD:
            temp.append(line)
    lines = temp

def generator(samples, batch_size=BATCH_SIZE):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            steering_angles = []
            for sample in batch_samples:
                center_img_path = sample[0].strip()
                left_img_path = sample[1].strip()
                right_img_path = sample[2].strip()

                steering_center = float(sample[3].strip())
                steering_left = steering_center + STEERING_CORRECTION
                steering_right = steering_center - STEERING_CORRECTION

                center_img = cv2.cvtColor(cv2.imread(center_img_path), cv2.COLOR_BGR2RGB)
                left_img = cv2.cvtColor(cv2.imread(left_img_path), cv2.COLOR_BGR2RGB)
                right_img = cv2.cvtColor(cv2.imread(right_img_path), cv2.COLOR_BGR2RGB)

                if NORMALIZE_BRIGHTNESS:
                    center_img = normalize_brightness(center_img)
                    left_img = normalize_brightness(left_img)
                    right_img = normalize_brightness(right_img)

                if ADD_FLIPS:
                    center_img_flipped = np.fliplr(center_img)
                    left_img_flipped = np.fliplr(left_img)
                    right_img_flipped = np.fliplr(right_img)

                    steering_center_flipped = -1.0 * steering_center
                    steering_left_flipped = -1.0 * steering_left
                    steering_right_flipped = -1.0 * steering_right

                images.append(center_img)
                steering_angles.append(steering_center)

                if ADD_SIDE_VIEWS:
                    images.extend([left_img, right_img])
                    steering_angles.extend([steering_left, steering_right])

                if ADD_FLIPS:
                    images.extend([center_img_flipped, left_img_flipped, right_img_flipped])
                    steering_angles.extend([steering_center_flipped, steering_left_flipped, steering_right_flipped])

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(steering_angles)
            yield shuffle(X_train, y_train)

def vgg16Model(input_shape):
    inputs = Input(shape=input_shape)

    x = Lambda(lambda img: (img / 255.0) - 0.5)(inputs)
    x = Cropping2D(cropping=((55, 25), (0, 0)))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    x = Flatten(name='flatten')(x)
    # x = Dense(100, activation='relu', name='fc1')(x)
    # x = Dense(50, activation='relu', name='fc2')(x)
    x = Dense(100, activation='relu', name='fc3')(x)
    output = Dense(1, name='output_layer')(x)

    model = Model(inputs=inputs, outputs=output)
    return model

def nvidiaModel(input_shape):
    inputs = Input(shape=input_shape)

    x = Lambda(lambda img: (img / 255.0) - 0.5)(inputs)
    x = Cropping2D(cropping=((70, 25), (0, 0)))(x)
    x = Conv2D(24, (5, 5), strides=(2, 2), activation='relu', name='conv1')(inputs)
    x = Conv2D(36, (5, 5), strides=(2, 2), activation='relu', name='conv2')(x)
    x = Conv2D(48, (5, 5), strides=(2, 2), activation='relu', name='conv3')(x)
    x = Conv2D(64, (3, 3), activation='relu', name='conv4')(x)
    x = Conv2D(64, (3, 3), activation='relu', name='conv5')(x)

    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)
    x = Dense(100, activation='relu', name='fc1')(x)
    #x = Dropout(0.5)(x)
    x = Dense(50, activation='relu', name='fc2')(x)
    #x = Dropout(0.5)(x)
    x = Dense(10, activation='relu', name='fc3')(x)
    #x = Dropout(0.5)(x)
    output = Dense(1, name='output_layer')(x)

    model = Model(inputs=inputs, outputs=output)
    return model

# Split data in to training and validaiton sets
train_samples, validation_samples = train_test_split(lines, test_size=0.2, random_state=42)

print('Train_samples.shape', len(train_samples))
print('Validation_samples.shape', len(validation_samples))

# Training new model
ts = str(int(time.time()))
model_name = 'nvidia'
num_epochs = 50
steps_per_epoch = int(len(train_samples)/BATCH_SIZE)
run_name = 'model={}-batch_size={}-num_epoch={}-steps_per_epoch={}-run-{}-ts={}'.format(model_name,
                                                                          BATCH_SIZE,
                                                                          num_epochs,
                                                                          steps_per_epoch,
                                                                          runs,
                                                                          ts)
print(run_name)
tensorboard_loc = os.path.join(TENSORBOARD_PATH, run_name)
checkpoint_loc = os.path.join(MODELS_PATH, 'model-{}.h5'.format(ts))

earlyStopping = EarlyStopping(monitor='val_loss',
                              patience=3,
                              verbose=1,
                              min_delta = 0.0001,
                              mode='min')

modelCheckpoint = ModelCheckpoint(checkpoint_loc,
                                  monitor = 'val_loss',
                                  save_best_only = True,
                                  mode = 'min',
                                  verbose = 1)

tensorboard = TensorBoard(log_dir=tensorboard_loc, histogram_freq=0, write_graph=True, write_images=True)
callbacks_list = [modelCheckpoint, earlyStopping, tensorboard]

train_generator = generator(train_samples)
validation_generator = generator(validation_samples)

model = nvidiaModel(IMG_SHAPE)
#model = load_model(os.path.join(MODELS_PATH, 'model-1511478745.h5'))
optim = Adam(lr=0.0001)
model.compile(loss='mse', optimizer=optim)
print(model.summary())

model.fit_generator(train_generator,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=validation_generator,
                    validation_steps=len(validation_samples),
                    verbose=1,
                    callbacks=callbacks_list,
                    epochs=num_epochs)

#model.save(checkpoint_loc)
print('Done training {}'.format(run_name))
