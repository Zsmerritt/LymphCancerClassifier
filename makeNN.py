from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization, LeakyReLU, ELU
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import initializers
from copy import deepcopy
import imageMod

#ensuring reproducable results
#THIS SHOULD BE REMOVED DURING FINAL TRAINING
import numpy as np
import tensorflow as tf
import random as rn

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


# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=180,
        width_shift_range=0.2,
        height_shift_range=0.2,
        fill_mode='nearest')


# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

trainDataLen=128909+87757
validDataLen=2000+1361

trainDataLenP=2000
validDataLenP=512

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
def trainGenerator(size, batch):
	return train_datagen.flow_from_directory(
        './data/prototyping/train/',  # this is the target directory
        target_size=(size, size),  # all images will be resized to 150x150
        batch_size=batch,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

def validationGenerator(size, batch):
# this is a similar generator, for validation data
	return test_datagen.flow_from_directory(
        './data/prototyping/valid/',
        target_size=(size, size),
        batch_size=batch,
        class_mode='binary')

def trainAndSave(model,epochs,name,image_size):
	#hold on to best model to save after training
	bestModel=model
	bestModelLoss,bestModelAcc=1.0,0.0
	try:
		#moved these out of loop to solve mem alocation prob
		initBatchSize=16
		trainGen=trainGenerator(image_size,initBatchSize)
		validGen=validationGenerator(image_size,initBatchSize)

		for x in range(1,epochs+1):
			#update batch_size and update generators when_batch size changes
			batch_size=calBatchSize(x,epochs)
			if batch_size!=initBatchSize:
				initBatchSize=batch_size
				trainGen=trainGenerator(image_size,batch_size)
				validGen=validationGenerator(image_size,batch_size)
			#print info and start epoch
			print('MODEL: ',name,' CURRENT EPOCH:',x)
			hist=model.fit_generator(
			        trainGen,
			        #steps_per_epoch=trainDataLen // batch_size,
			        steps_per_epoch=trainDataLenP // batch_size,
			        epochs=1,
			        validation_data=validGen,
			        #validation_steps=validDataLen // batch_size,
			        validation_steps=validDataLenP // batch_size,
			        verbose=1,
			        max_queue_size=16)
			print(hist.history)
			#cal loss and accuracy before comparing to previous best model
			#loss,acc=model.evaluate_generator(validGen)
			loss,acc=model.evaluate_generator(validGen)
			if bestModelAcc<acc and bestModelLoss>loss:
				bestModel=deepcopy(model)
				bestModelLoss,bestModelAcc=loss,acc
		#save best model created
		bestModel.save_weights('./weights/weights_'+name+'_'+str(round(bestModelAcc,5))+'.h5')
		bestModel.save('./models/model_'+name+'_'+str(round(bestModelAcc,5))+'.dnn') 
	except KeyboardInterrupt as e:
		print('Saving best model generated so far')
		bestModel.save_weights('./weights/weights_'+name+'_'+str(round(bestModelAcc,5))+'.h5')
		bestModel.save('./models/model_'+name+'_'+str(round(bestModelAcc,5))+'.dnn') 
		raise KeyboardInterrupt


def calBatchSize(epoch, totalEpochs):
	if epoch<=totalEpochs//6:
		return 16
	elif epoch<=(totalEpochs//6)*2:
		return 32
	elif epoch<=(totalEpochs//6)*3:
		return 64
	elif epoch<=(totalEpochs//6)*4:
		return 128
	elif epoch <=(totalEpochs//6)*5:
		return 256
	else:
		return 512

#~85% max, removed 256 dense layer from bottom
def model1():

	dropout=0.3
	kernel_size=(3,3)
	pool_size=(2,2)
	image_size=200
	epochs=50
	name='prototype1'

	model = Sequential()
	model.add(Conv2D(32, kernel_size=kernel_size, kernel_initializer=initializers.he_normal(seed=None), input_shape=(image_size, image_size, 3)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(32, kernel_size=kernel_size, kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(64, kernel_size=kernel_size, kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(64, kernel_size=kernel_size, kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(128, kernel_size=kernel_size, kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(256, kernel_size=kernel_size, kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

	model.add(Dense(128, kernel_initializer=initializers.lecun_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(Dropout(dropout))

	model.add(Dense(64, kernel_initializer=initializers.lecun_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(Dropout(dropout))

	model.add(Dense(32, kernel_initializer=initializers.lecun_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(Dropout(dropout))

	model.add(Dense(16, kernel_initializer=initializers.lecun_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(Dropout(dropout))
				
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	model.compile(loss='binary_crossentropy',
	              optimizer='adam',
	              metrics=['accuracy'])

	trainAndSave(model,epochs,name,image_size)

#leakyReLU instead of RELU
def model2():

	dropout=0.3
	kernel_size=(3,3)
	pool_size=(2,2)
	image_size=200
	epochs=50
	name='prototype2'

	model = Sequential()
	model.add(Conv2D(32, kernel_size=kernel_size, kernel_initializer=initializers.he_normal(seed=None), input_shape=(image_size, image_size, 3)))
	model.add(LeakyReLU(alpha=0.3))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(32, kernel_size=kernel_size, kernel_initializer=initializers.he_normal(seed=None)))
	model.add(LeakyReLU(alpha=0.3))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(64, kernel_size=kernel_size, kernel_initializer=initializers.he_normal(seed=None)))
	model.add(LeakyReLU(alpha=0.3))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(64, kernel_size=kernel_size, kernel_initializer=initializers.he_normal(seed=None)))
	model.add(LeakyReLU(alpha=0.3))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(128, kernel_size=kernel_size, kernel_initializer=initializers.he_normal(seed=None)))
	model.add(LeakyReLU(alpha=0.3))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(256, kernel_size=kernel_size, kernel_initializer=initializers.he_normal(seed=None)))
	model.add(LeakyReLU(alpha=0.3))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

	model.add(Dense(128, kernel_initializer=initializers.lecun_normal(seed=None)))
	model.add(LeakyReLU(alpha=0.3))
	model.add(Dropout(dropout))

	model.add(Dense(64, kernel_initializer=initializers.lecun_normal(seed=None)))
	model.add(LeakyReLU(alpha=0.3))
	model.add(Dropout(dropout))

	model.add(Dense(32, kernel_initializer=initializers.lecun_normal(seed=None)))
	model.add(LeakyReLU(alpha=0.3))
	model.add(Dropout(dropout))

	model.add(Dense(16, kernel_initializer=initializers.lecun_normal(seed=None)))
	model.add(LeakyReLU(alpha=0.3))
	model.add(Dropout(dropout))
				
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	model.compile(loss='binary_crossentropy',
	              optimizer='adam',
	              metrics=['accuracy'])

	trainAndSave(model,epochs,name,image_size)


#increased kernal size to (5,5) from (3,3)
def model3():

	dropout=0.3
	kernel_size=(5,5)
	pool_size=(2,2)
	image_size=200
	epochs=50
	name='prototype3'

	model = Sequential()
	model.add(Conv2D(32, kernel_size=kernel_size, kernel_initializer=initializers.he_normal(seed=None), input_shape=(image_size, image_size, 3)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(32, kernel_size=kernel_size, kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(64, kernel_size=kernel_size, kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(64, kernel_size=kernel_size, kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(128, kernel_size=kernel_size, kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(256, kernel_size=kernel_size, kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

	model.add(Dense(128, kernel_initializer=initializers.lecun_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(Dropout(dropout))

	model.add(Dense(64, kernel_initializer=initializers.lecun_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(Dropout(dropout))

	model.add(Dense(32, kernel_initializer=initializers.lecun_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(Dropout(dropout))

	model.add(Dense(16, kernel_initializer=initializers.lecun_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(Dropout(dropout))
				
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	model.compile(loss='binary_crossentropy',
	              optimizer='adam',
	              metrics=['accuracy'])

	trainAndSave(model,epochs,name,image_size)


#added in stride values of (2,2) and post flatten dropout
def model4():
	#jumpOut = (featInit-featOut)/featOut-1  OR  stride*JumpIn
	#receptive field size = prevLayerRCF + (K-1) * jumpSize

	dropout=0.3
	strides=(2,2)
	kernel_size=(3,3)
	pool_size=(2,2)
	image_size=200
	epochs=50
	name='prototype4'

	model = Sequential()

	model.add(Conv2D(32, kernel_size=kernel_size, strides=strides, kernel_initializer=initializers.he_normal(seed=None), input_shape=(image_size, image_size, 3)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))
	#RFS = 1 + 2 * 2 = 5
	#c = 3/5 = 0.6

	model.add(Conv2D(32, kernel_size=kernel_size, strides=strides, kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))
	#RFS = 5 + 2 * 4 = 13
	#C = 3*2/13 = 0.46

	model.add(Conv2D(64, kernel_size=kernel_size, strides=strides, kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))
	#RFS = 13 + 2 * 8 = 29
	#C = 3*4 / 29 = 0.41

	model.add(Conv2D(64, kernel_size=kernel_size, strides=strides, kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))
	#RFS = 29 + 2 * 16 = 61
	#C = 3*8 / 61 = 0.39

	model.add(Conv2D(128, kernel_size=kernel_size, strides=strides, kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))
	#RFS = 61 + 2 * 32 = 125
	#C = 3*16 / 125 = 0.383

	model.add(Conv2D(128, kernel_size=kernel_size, strides=strides, kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))
	#RFS = 125 + 2 * 64 = 253
	#C = 3*32 / 253 = 0.38


	model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
	model.add(Dropout(dropout))

	model.add(Dense(128, kernel_initializer=initializers.lecun_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(Dropout(dropout))

	model.add(Dense(64, kernel_initializer=initializers.lecun_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(Dropout(dropout))

	model.add(Dense(32, kernel_initializer=initializers.lecun_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(Dropout(dropout))

	model.add(Dense(16, kernel_initializer=initializers.lecun_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(Dropout(dropout))
				
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	model.compile(loss='binary_crossentropy',
	              optimizer='adam',
	              metrics=['accuracy'])

	trainAndSave(model,epochs,name,image_size)


#method allows for restarting models which are trainging poorly
def modelStart(modelName):
	try:
		modelName()
		return True
	except KeyboardInterrupt as e:
		print('KeyboardInterrupt detected, ending training')
		return False


def main():
	while not modelStart(model1):
		if input('Would you like to restart this model? (y or n) ')=="n":
			break
	while not modelStart(model2):
		if input('Would you like to restart this model? (y or n) ')=="n":
			break
	while not modelStart(model3):
		if input('Would you like to restart this model? (y or n) ')=="n":
			break
	while not modelStart(model4):
		if input('Would you like to restart this model? (y or n) ')=="n":
			break




if __name__ == '__main__':
	main()