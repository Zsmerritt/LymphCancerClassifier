import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(allow_soft_placement=False)
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))



from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization, GaussianNoise
from keras import initializers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from data_generator import DataGenerator


#ensuring reproducable results
#THIS SHOULD BE REMOVED DURING FINAL TRAINING
import numpy as np
import random as rn

'''
# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.

np.random.seed(42)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

rn.seed(12345)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
							  inter_op_parallelism_threads=1)

from keras import backend as K

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/set_random_seed

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
'''

trainSetFolder='./data/train/'
validSetFolder='./data/valid/'

target_size=(96,96)

trainDataLen=128909+87757
validDataLen=2000+1361

trainDataLenP=2000
validDataLenP=512


def train_generator_with_batch_schedule(
						model,epochs,name,target_size,batch_size,
						model_save_filepath):

	epochs=epochs//3
	max_queue_size=20

	train_gen = DataGenerator(
		data_folder=trainSetFolder,
		rescale=1./255,
		horizontal_flip=True,
		vertical_flip=True,
		target_size=target_size,
		batch_size=batch_size,
		rotation_range=4)

	valid_gen = DataGenerator(
		data_folder=validSetFolder,
		rescale=1./255,
		target_size=target_size,
		batch_size=batch_size)

	for x in range(1,4):
		train_gen.update_batch_size(batch_size)
		valid_gen.update_batch_size(batch_size)
		model=trainAndSaveGenerator(
									model,epochs*x,name,target_size,batch_size,max_queue_size,
									model_save_filepath,epochs*(x-1),train_gen,valid_gen
									)
		batch_size=batch_size*2
		max_queue_size=max_queue_size//2

#using generator
def trainAndSaveGenerator(
						model,epochs,name,target_size,batch_size,
						max_queue_size,model_save_filepath,
						initial_epoch,train_gen,valid_gen):

	model.fit_generator(
		generator=train_gen,
		steps_per_epoch=trainDataLen // batch_size,
		#steps_per_epoch=trainDataLenP // batch_size,
		epochs=epochs,
		validation_data=valid_gen,
		validation_steps=validDataLen // batch_size,
		#validation_steps=validDataLenP // batch_size,
		verbose=1,
		max_queue_size=max_queue_size,
		use_multiprocessing=True,
		initial_epoch=initial_epoch,
		workers=4,
		callbacks=[
			EarlyStopping(patience=4, monitor='val_acc', restore_best_weights=True),
			ReduceLROnPlateau(patience=3,factor=0.4,min_lr=0.001),
			ModelCheckpoint(model_save_filepath, monitor='val_acc', save_best_only=True)
		])
	return model

'''
ideas for next run:
	massively increase kernal numbers, start with 128 in place of 32 and increase with the same schedule
	continue calculating receptive field size and ensure that image size does not become too much smaller than it
	try model2 again but with much larger kernal sizes, maybe drop the last conv layer as the RFS is quite a bit bigger than the image size
	impliment modelcheckpoint callback function
	increase learning rate factor to 0.4-0.7 to push past ~92% barrier

'''

#added stride, removed some conv2d and dropout layers, using nadam
def model1():

	dropout=0.3
	kernel_size=(5,5)
	pool_size=(2,2)
	image_size=96
	epochs=60
	name='model-1'
	max_queue_size=16
	batch_size=64
	stride=(2,2)
	filepath='./models/model-1/model-1.{val_acc:.3f}-{epoch:02d}.hdf5'



	model = Sequential()
	model.add(GaussianNoise(0.1,input_shape=(image_size, image_size, 3)))
	model.add(Conv2D(128, kernel_size=kernel_size, padding="same", strides=stride, kernel_initializer=initializers.he_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	#model.add(Dropout(dropout))
	#RFS= 1 + 2*1 = 3

	model.add(Conv2D(128, kernel_size=kernel_size, padding="same", strides=stride, kernel_initializer=initializers.he_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	#model.add(Dropout(dropout))
	#RFS = 3 + 2 * 2 = 7

	model.add(Conv2D(256, kernel_size=kernel_size, padding="same", strides=stride, kernel_initializer=initializers.he_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	#model.add(Dropout(dropout))
	#RFS = 7 + 2 * 4 = 15


	model.add(MaxPooling2D(pool_size=pool_size))
	#model.add(Dropout(0.7))


	model.add(Conv2D(256, kernel_size=kernel_size, padding="same", strides=stride, kernel_initializer=initializers.he_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	#model.add(Dropout(dropout))
	#RFS = 15 + 2 * 8 = 31

	model.add(Conv2D(512, kernel_size=kernel_size, padding="same", strides=stride, kernel_initializer=initializers.he_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	#model.add(Dropout(dropout))
	#RFS = 31 + 2 * 16 = 63

	model.add(MaxPooling2D(pool_size=pool_size))

	model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

	model.add(Dense(128, kernel_initializer=initializers.lecun_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	#model.add(Dropout(dropout))

	model.add(Dense(64, kernel_initializer=initializers.lecun_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	#model.add(Dropout(dropout))

	model.add(Dense(32, kernel_initializer=initializers.lecun_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	#model.add(Dropout(dropout))

	model.add(Dense(16, kernel_initializer=initializers.lecun_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	#model.add(Dropout(dropout))
				
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	model.compile(loss='binary_crossentropy',
				  optimizer='nadam',
				  metrics=['accuracy'])

	train_generator_with_batch_schedule(model,epochs,name,target_size,batch_size,filepath)


#added stride, removed some conv2d and dropout layers
def model2():

	dropout=0.3
	kernel_size=(5,5)
	pool_size=(2,2)
	image_size=96
	epochs=60
	name='model-2'
	max_queue_size=16
	batch_size=64
	stride=(2,2)
	filepath='./models/model-2/model-2.{val_acc:.3f}-{epoch:02d}.hdf5'



	model = Sequential()
	model.add(GaussianNoise(0.1,input_shape=(image_size, image_size, 3)))
	model.add(Conv2D(128, kernel_size=kernel_size, padding="same", strides=stride, kernel_initializer=initializers.he_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	#model.add(Dropout(dropout))
	#RFS= 1 + 2*1 = 3

	model.add(Conv2D(128, kernel_size=kernel_size, padding="same", strides=stride, kernel_initializer=initializers.he_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	#model.add(Dropout(dropout))
	#RFS = 3 + 2 * 2 = 7

	model.add(Conv2D(256, kernel_size=kernel_size, padding="same", strides=stride, kernel_initializer=initializers.he_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	#model.add(Dropout(dropout))
	#RFS = 7 + 2 * 4 = 15


	model.add(MaxPooling2D(pool_size=pool_size))
	#model.add(Dropout(0.7))


	model.add(Conv2D(256, kernel_size=kernel_size, padding="same", strides=stride, kernel_initializer=initializers.he_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	#model.add(Dropout(dropout))
	#RFS = 15 + 2 * 8 = 31

	model.add(Conv2D(512, kernel_size=kernel_size, padding="same", strides=stride, kernel_initializer=initializers.he_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	#model.add(Dropout(dropout))
	#RFS = 31 + 2 * 16 = 63

	model.add(MaxPooling2D(pool_size=pool_size))

	model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

	model.add(Dense(128, kernel_initializer=initializers.lecun_normal()))
	model.add(Activation('relu'))
	model.add(Dropout(dropout))

	model.add(Dense(64, kernel_initializer=initializers.lecun_normal()))
	model.add(Activation('relu'))
	#model.add(Dropout(dropout))

	model.add(Dense(32, kernel_initializer=initializers.lecun_normal()))
	model.add(Activation('relu'))
	model.add(Dropout(dropout))

	model.add(Dense(16, kernel_initializer=initializers.lecun_normal()))
	model.add(Activation('relu'))
	#model.add(Dropout(dropout))
				
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	model.compile(loss='binary_crossentropy',
				  optimizer='adam',
				  metrics=['accuracy'])

	train_generator_with_batch_schedule(model,epochs,name,target_size,batch_size,filepath)


#removed a max pooling layers, removed all dropout, added noise layer at beginning 
def model3():

	dropout=0.3
	kernel_size=(5,5)
	pool_size=(2,2)
	image_size=96
	epochs=60
	name='model-3'
	max_queue_size=16
	batch_size=64
	filepath='./models/model-3/model-3.{val_acc:.3f}-{epoch:02d}.hdf5'


	model = Sequential()
	model.add(GaussianNoise(0.1,input_shape=(image_size, image_size, 3)))
	model.add(Conv2D(128, kernel_size=kernel_size, padding="same", kernel_initializer=initializers.he_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))

	model.add(Conv2D(128, kernel_size=kernel_size, padding="same", kernel_initializer=initializers.he_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))

	model.add(Conv2D(256, kernel_size=kernel_size, padding="same", kernel_initializer=initializers.he_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	#model.add(MaxPooling2D(pool_size=pool_size))

	model.add(Conv2D(256, kernel_size=kernel_size, padding="same", kernel_initializer=initializers.he_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))

	model.add(Conv2D(512, kernel_size=kernel_size, padding="same", kernel_initializer=initializers.he_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	#model.add(Dropout(dropout))

	model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

	model.add(Dense(128, kernel_initializer=initializers.lecun_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	#model.add(Dropout(dropout))

	model.add(Dense(64, kernel_initializer=initializers.lecun_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))

	#model.add(Dropout(dropout))

	model.add(Dense(32, kernel_initializer=initializers.lecun_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	#model.add(Dropout(dropout))

	model.add(Dense(16, kernel_initializer=initializers.lecun_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	#model.add(Dropout(dropout))
				
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	model.compile(loss='binary_crossentropy',
				  optimizer='nadam',
				  metrics=['accuracy'])

	train_generator_with_batch_schedule(model,epochs,name,target_size,batch_size,filepath)


#changed number of kernals (96), removed all droupout, added noise layer at beginning
def model4():

	dropout=0.8
	kernel_size=(5,5)
	pool_size=(2,2)
	image_size=96
	epochs=60
	name='model-4'
	batch_size=64
	filepath='./models/model-4/model-4.{val_acc:.3f}-{epoch:02d}.hdf5'


	model = Sequential()
	model.add(GaussianNoise(0.3,input_shape=(image_size, image_size, 3)))
	model.add(Conv2D(128, kernel_size=kernel_size, padding="same", kernel_initializer=initializers.he_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))

	model.add(GaussianNoise(0.3))
	model.add(Conv2D(128, kernel_size=kernel_size, padding="same", kernel_initializer=initializers.he_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))

	model.add(GaussianNoise(0.3))
	model.add(Conv2D(256, kernel_size=kernel_size, padding="same", kernel_initializer=initializers.he_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	#model.add(MaxPooling2D(pool_size=pool_size))

	model.add(GaussianNoise(0.3))
	model.add(Conv2D(256, kernel_size=kernel_size, padding="same", kernel_initializer=initializers.he_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))

	model.add(GaussianNoise(0.3))
	model.add(Conv2D(512, kernel_size=kernel_size, padding="same", kernel_initializer=initializers.he_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(0.6))

	model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

	model.add(GaussianNoise(0.3))
	model.add(Dense(128, kernel_initializer=initializers.lecun_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(Dropout(dropout))

	model.add(GaussianNoise(0.3))
	model.add(Dense(64, kernel_initializer=initializers.lecun_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(Dropout(dropout))

	model.add(GaussianNoise(0.3))
	model.add(Dense(32, kernel_initializer=initializers.lecun_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(Dropout(dropout))

	model.add(GaussianNoise(0.3))
	model.add(Dense(16, kernel_initializer=initializers.lecun_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(Dropout(dropout))
				
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	model.compile(loss='binary_crossentropy',
				  optimizer='adam',
				  metrics=['accuracy'])

	train_generator_with_batch_schedule(model,epochs,name,target_size,batch_size,filepath)



#method allows for restarting models which are trainging poorly
def modelStart(modelName):
	try:
		modelName()
		return True
	except KeyboardInterrupt as e:
		print('KeyboardInterrupt detected, ending training')
		return False


def main():
	while not modelStart(model4):
		if input('Would you like to restart this model? (y or n) ')=="n":
			break
	while not modelStart(model3):
		if input('Would you like to restart this model? (y or n) ')=="n":
			break
	while not modelStart(model2):
		if input('Would you like to restart this model? (y or n) ')=="n":
			break
	while not modelStart(model1):
		if input('Would you like to restart this model? (y or n) ')=="n":
			break
	


if __name__ == '__main__':
	main()
