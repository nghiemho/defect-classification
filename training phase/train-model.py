# import all packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras import optimizers
# create model architect
classifier = Sequential()
classifier.add(Convolution2D(32, 5, 5, input_shape = (192, 192, 1), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Convolution2D(32, 5, 5, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(output_dim = 64, activation = 'relu'))
classifier.add(Dense(output_dim = 6, activation = 'softmax'))
optimizer = optimizers.Adam(lr=0.0001)
classifier.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
# preprocessing the input for network
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                  vertical_flip = True,
                                  rotation_range = 20)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('/content/gdrive/My Drive/Python/Lasor-dataset/Training-set',
                                                 target_size = (192, 192),
                                                 batch_size = 5,
                                                 color_mode = 'grayscale',
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('/content/gdrive/My Drive/Python/Lasor-dataset/Test-set',
                                            target_size = (192, 192),
                                            batch_size = 5,
                                            color_mode = 'grayscale',
                                            class_mode = 'categorical')
# feed to model
classifier.fit_generator(training_set,
                         samples_per_epoch = 270,
                         nb_epoch = 30,
                         validation_data = test_set,
                         nb_val_samples = 90
                        
                        )