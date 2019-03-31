from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization, LeakyReLU, ELU
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import initializers
from copy import deepcopy
import dataGen


#ensuring reproducable results
#THIS SHOULD BE REMOVED DURING FINAL TRAINING
import numpy as np
import tensorflow as tf
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

# this is the augmentation configuration we will use for training
train_transform_map = dataGen.get_transform_map(
        data_folder=trainSetFolder,
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        contrast_stretching=True, 
		histogram_equalization=False, 
		adaptive_equalization=True,
		brightness_range=(0.0,1.5),
		zca_whitening=True)


# this is the augmentation configuration we will use for testing:
# only rescaling
valid_transform_map = dataGen.get_transform_map(data_folder=validSetFolder,
											rescale=1./255)

target_size=(96,96)
'''
print('generating data')
train=dataGen.image_processor(transform_map=train_transform_map,target_size=target_size)
valid=dataGen.image_processor(transform_map=valid_transform_map,target_size=target_size)
print('finished processing data')
'''
trainDataLen=128909+87757
validDataLen=2000+1361

trainDataLenP=2000
validDataLenP=512


'''
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

'''

def shuffleData(data_dict):
	perm=np.random.permutation(data_dict['data'].shape[0])
	data_dict['data'],data_dict['labels']=data_dict['data'][perm],data_dict['labels'][perm]

#using generator
def trainAndSaveGenerator(model,epochs,name,target_size):
	#hold on to best model to save after training
	bestModel=model
	bestModelLoss,bestModelAcc=1.0,0.0
	try:
		for x in range(0,epochs):
			#update batch_size 
			batch_size=calBatchSize(x,epochs)

			#print info and start epoch
			print('MODEL: ',name,' CURRENT EPOCH:',x+1)
			hist=model.fit_generator(
			        dataGen.image_generator(transform_map=train_transform_map,target_size=(target_size,target_size),batch_size=batch_size),
			        #steps_per_epoch=trainDataLen // batch_size,
			        steps_per_epoch=trainDataLenP // batch_size,
			        epochs=1,
			        validation_data=dataGen.image_generator(transform_map=valid_transform_map,target_size=(target_size,target_size),batch_size=batch_size),
			        #validation_steps=validDataLen // batch_size,
			        validation_steps=validDataLenP // batch_size,
			        verbose=1,
			        max_queue_size=16)
			#cal loss and accuracy before comparing to previous best model
			acc,loss=hist.history['val_acc'][0],hist.history['val_loss'][0]
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

def trainAndSave(model,epochs,name):
	#hold on to best model to save after training
	bestModel=model
	bestModelLoss,bestModelAcc=1.0,0.0

	try:
		for x in range(0,epochs):
			#shuffle data to normalize
			shuffleData(train)
			#update batch_size 
			batch_size=calBatchSize(x+1,epochs)
			#print info and start epoch
			print('MODEL: '+str(name)+'  CURRENT EPOCH: '+str(x+1)+"/"+str(epochs)+'  BATCH SIZE: '+str(batch_size))
			hist=model.fit(
			        x=train['data'],
			        y=train['labels'],
			        batch_size=batch_size,
			        epochs=1,
			        verbose=1,
			        validation_data=(valid['data'],valid['labels']))

			#cal loss and accuracy before comparing to previous best model
			acc,loss=hist.history['val_acc'][0],hist.history['val_loss'][0]
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

def trainAndSaveBatch(model,epochs,name,target_size):
	#hold on to best model to save after training
	bestModel=model
	bestModelLoss,bestModelAcc=1.0,0.0


	try:
		for x in range(0,epochs):
			#update batch_size 
			batch_size=calBatchSize(x+1,epochs)
			steps_per_epoch_train=trainDataLen//batch_size
			epoch_desc='MODEL: '+str(name)+'  CURRENT EPOCH: '+str(x+1)+"/"+str(epochs)+'  BATCH SIZE: '+str(batch_size)
			for y in tqdm(range(steps_per_epoch_train), desc=epoch_desc):

				train=dataGen.image_processor_batch(transform_map=train_transform_map,target_size=target_size,batch_size=batch_size)
				model.train_on_batch(
			        x=train['data'],
			        y=train['labels'])
			'''
			#cal loss and accuracy before comparing to previous best model
			acc = model.evaluate(
							x=valid['data'],
							y=valid['labels'],
							batch_size=batch_size,
							verbose=1)
							#['val_acc'][0],hist.history['val_loss'][0]
			'''
			acc=test_model_accuracy(model=model,transform_map=valid_transform_map,target_size=target_size,batch_size=batch_size)
			print("Model Validation Accuracy: ",acc)
			if bestModelAcc<acc:
				bestModel=deepcopy(model)
				bestModelAcc=acc
		#save best model created
		bestModel.save_weights('./weights/weights_'+name+'_'+str(round(bestModelAcc,5))+'.h5')
		bestModel.save('./models/model_'+name+'_'+str(round(bestModelAcc,5))+'.dnn') 
	except KeyboardInterrupt as e:
		print('Saving best model generated so far')
		bestModel.save_weights('./weights/weights_'+name+'_'+str(round(bestModelAcc,5))+'.h5')
		bestModel.save('./models/model_'+name+'_'+str(round(bestModelAcc,5))+'.dnn') 
		raise KeyboardInterrupt

def test_model_accuracy(model,transform_map,target_size,batch_size):
	correct=0
	for x in tqdm(range(validDataLen),desc='Evaluating Model 		'):
		valid=dataGen.image_processor_batch(transform_map=transform_map,target_size=target_size,batch_size=1)
		prediction=round(model.predict(valid['data'].reshape(1, target_size[0], target_size[1], 3))[0][0])
		if prediction== valid['labels'][0]:correct+=1
	return correct/validDataLen



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

def model1():

	dropout=0.3
	kernel_size=(5,5)
	pool_size=(2,2)
	image_size=96
	epochs=50
	name='prototype1'

	model = Sequential()
	model.add(Conv2D(32, kernel_size=kernel_size, padding="same", kernel_initializer=initializers.he_normal(), input_shape=(image_size, image_size, 3)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(32, kernel_size=kernel_size, padding="same", kernel_initializer=initializers.he_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(64, kernel_size=kernel_size, padding="same", kernel_initializer=initializers.he_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(64, kernel_size=kernel_size, padding="same", kernel_initializer=initializers.he_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(128, kernel_size=kernel_size, padding="same", kernel_initializer=initializers.he_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

	model.add(Dense(128, kernel_initializer=initializers.lecun_normal()))
	model.add(Activation('relu'))
	model.add(Dropout(dropout))

	model.add(Dense(64, kernel_initializer=initializers.lecun_normal()))
	model.add(Activation('relu'))
	model.add(Dropout(dropout))

	model.add(Dense(32, kernel_initializer=initializers.lecun_normal()))
	model.add(Activation('relu'))
	model.add(Dropout(dropout))

	model.add(Dense(16, kernel_initializer=initializers.lecun_normal()))
	model.add(Activation('relu'))
	model.add(Dropout(dropout))
				
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	model.compile(loss='binary_crossentropy',
	              optimizer='adam',
	              metrics=['accuracy'])

	trainAndSaveBatch(model,epochs,name,(image_size,image_size))


#added 256 dense layer
def model2():

	dropout=0.3
	kernel_size=(5,5)
	pool_size=(2,2)
	image_size=96
	epochs=50
	name='prototype2'

	model = Sequential()
	model.add(Conv2D(32, kernel_size=kernel_size, padding="same", kernel_initializer=initializers.he_normal(), input_shape=(image_size, image_size, 3)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(32, kernel_size=kernel_size, padding="same", kernel_initializer=initializers.he_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(64, kernel_size=kernel_size, padding="same", kernel_initializer=initializers.he_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(64, kernel_size=kernel_size, padding="same", kernel_initializer=initializers.he_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(128, kernel_size=kernel_size, padding="same", kernel_initializer=initializers.he_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

	model.add(Dense(256, kernel_initializer=initializers.lecun_normal()))
	model.add(Activation('relu'))
	model.add(Dropout(dropout))

	model.add(Dense(128, kernel_initializer=initializers.lecun_normal()))
	model.add(Activation('relu'))
	model.add(Dropout(dropout))

	model.add(Dense(64, kernel_initializer=initializers.lecun_normal()))
	model.add(Activation('relu'))
	model.add(Dropout(dropout))

	model.add(Dense(32, kernel_initializer=initializers.lecun_normal()))
	model.add(Activation('relu'))
	model.add(Dropout(dropout))

	model.add(Dense(16, kernel_initializer=initializers.lecun_normal()))
	model.add(Activation('relu'))
	model.add(Dropout(dropout))
				
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	model.compile(loss='binary_crossentropy',
	              optimizer='adam',
	              metrics=['accuracy'])

	trainAndSaveBatch(model,epochs,name,(image_size,image_size))


#increased dropout to 0.7 and removed many dropout layers
def model3():

	dropout=0.7
	kernel_size=(5,5)
	pool_size=(2,2)
	image_size=96
	epochs=50
	name='prototype3'

	model = Sequential()
	model.add(Conv2D(32, kernel_size=kernel_size, padding="same", kernel_initializer=initializers.he_normal(), input_shape=(image_size, image_size, 3)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))

	model.add(Conv2D(32, kernel_size=kernel_size, padding="same", kernel_initializer=initializers.he_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))

	model.add(Conv2D(64, kernel_size=kernel_size, padding="same", kernel_initializer=initializers.he_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(64, kernel_size=kernel_size, padding="same", kernel_initializer=initializers.he_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))

	model.add(Conv2D(128, kernel_size=kernel_size, padding="same", kernel_initializer=initializers.he_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(0.2))

	model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

	model.add(Dense(256, kernel_initializer=initializers.lecun_normal()))
	model.add(Activation('relu'))

	model.add(Dense(128, kernel_initializer=initializers.lecun_normal()))
	model.add(Activation('relu'))

	model.add(Dense(64, kernel_initializer=initializers.lecun_normal()))
	model.add(Activation('relu'))
	model.add(Dropout(dropout))

	model.add(Dense(32, kernel_initializer=initializers.lecun_normal()))
	model.add(Activation('relu'))

	model.add(Dense(16, kernel_initializer=initializers.lecun_normal()))
	model.add(Activation('relu'))
				
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	model.compile(loss='binary_crossentropy',
	              optimizer='adam',
	              metrics=['accuracy'])

	trainAndSaveBatch(model,epochs,name,(image_size,image_size))


#changed number of kernals (81-84)
def model4():

	dropout=0.3
	kernel_size=(5,5)
	pool_size=(2,2)
	image_size=96
	epochs=50
	name='prototype4'

	model = Sequential()
	model.add(Conv2D(32, kernel_size=kernel_size, padding="same", kernel_initializer=initializers.he_normal(), input_shape=(image_size, image_size, 3)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(64, kernel_size=kernel_size, padding="same", kernel_initializer=initializers.he_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(128, kernel_size=kernel_size, padding="same", kernel_initializer=initializers.he_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(256, kernel_size=kernel_size, padding="same", kernel_initializer=initializers.he_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(512, kernel_size=kernel_size, padding="same", kernel_initializer=initializers.he_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

	model.add(Dense(128, kernel_initializer=initializers.lecun_normal()))
	model.add(Activation('relu'))
	model.add(Dropout(dropout))

	model.add(Dense(64, kernel_initializer=initializers.lecun_normal()))
	model.add(Activation('relu'))
	model.add(Dropout(dropout))

	model.add(Dense(32, kernel_initializer=initializers.lecun_normal()))
	model.add(Activation('relu'))
	model.add(Dropout(dropout))

	model.add(Dense(16, kernel_initializer=initializers.lecun_normal()))
	model.add(Activation('relu'))
	model.add(Dropout(dropout))
				
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	model.compile(loss='binary_crossentropy',
	              optimizer='adam',
	              metrics=['accuracy'])

	trainAndSaveBatch(model,epochs,name,(image_size,image_size))




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