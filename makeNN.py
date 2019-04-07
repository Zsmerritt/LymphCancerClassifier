import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(allow_soft_placement=False)
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))



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
from tqdm import tqdm
from threading import Thread
import time
from data_loader import data_loader, data_loader_generator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

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
		brightness_range=(0.0,1.5))
		#zca_whitening=True)


# this is the augmentation configuration we will use for testing:
# only rescaling
valid_transform_map = dataGen.get_transform_map(data_folder=validSetFolder,
												rescale=1./255)
train_datagen = ImageDataGenerator(
		rescale=1./255,
		horizontal_flip=True,
		vertical_flip=True,
		fill_mode='nearest',
		brightness_range=(0.0,1.5))
		#zca_whitening=True)


# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

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



# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
def trainGenerator(size, batch):
	return train_datagen.flow_from_directory(
		trainSetFolder,  # this is the target directory
		target_size=size,  # all images will be resized to 150x150
		batch_size=batch,
		class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

def validationGenerator(size, batch):
# this is a similar generator, for validation data
	return test_datagen.flow_from_directory(
		validSetFolder,
		target_size=size,
		batch_size=batch,
		class_mode='binary')




def train_model(model,epochs,name,target_size,train_transform_map,valid_transform_map,max_queue_size):
	train_data_loader=data_loader(train_transform_map,target_size,32,max_queue_size)
	valid_data_loader=data_loader(valid_transform_map,target_size,32,max_queue_size)

	train_thread=Thread(data_loader_generator(train_data_loader))
	valid_thread=Thread(data_loader_generator(valid_data_loader))
	time.sleep(10)
	Thread(trainAndSaveBatch(model,epochs,name,target_size,train_data_loader,valid_data_loader))

	train_data_loader_term=train_data_loader.get_terminate()
	valid_data_loader_term=valid_data_loader.get_terminate()
	while train_data_loader_term or valid_data_loader_term:
		if train_data_loader.get_terminate():
			Thread.join(train_thread)
		if valid_data_loader.get_terminate():
			Thread.join(valid_thread)



def shuffleData(data_dict):
	perm=np.random.permutation(data_dict['data'].shape[0])
	data_dict['data'],data_dict['labels']=data_dict['data'][perm],data_dict['labels'][perm]

#using generator
def trainAndSaveGenerator(model,epochs,name,target_size,batch_size,model_save_filepath):
	#hold on to best model to save after training
	trainGen=trainGenerator(target_size,batch_size)
	validGen=validationGenerator(target_size,batch_size)
	#print info and start epoch
	hist=model.fit_generator(
			trainGen,
			steps_per_epoch=trainDataLen // batch_size,
			#steps_per_epoch=trainDataLenP // batch_size,
			epochs=epochs,
			validation_data=validGen,
			validation_steps=validDataLen // batch_size,
			#validation_steps=validDataLenP // batch_size,
			verbose=1,
			max_queue_size=16,
			use_multiprocessing=True,
			workers=8,
			callbacks=[
				EarlyStopping(patience=8, restore_best_weights=True),
				ReduceLROnPlateau(patience=3,factor=0.7,min_lr=0.001),
				ModelCheckpoint(model_save_filepath, monitor='val_loss', save_best_only=True)

			])

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
					validation_data=(valid['data'],valid['labels']),
					use_multiprocessing=True,
					workers=8)

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
	except MemoryError as e:
		print('Memory Error! Saving best model generated so far')
		bestModel.save_weights('./weights/weights_'+name+'_'+str(round(bestModelAcc,5))+'.h5')
		bestModel.save('./models/model_'+name+'_'+str(round(bestModelAcc,5))+'.dnn')

def trainAndSaveBatch(model,epochs,name,target_size,train_data_loader,valid_data_loader):
	#hold on to best model to save after training
	bestModel=model
	bestModelLoss,bestModelAcc=1.0,0.0

	batch_size=32
	try:
		for x in range(0,epochs):
			#update batch_size 
			if calBatchSize(x+1,epochs)!=batch_size:
				batch_size=calBatchSize(x+1,epochs)
				train_data_loader.update_batch_size(batch_size)

			steps_per_epoch_train=2#trainDataLen//batch_size
			epoch_desc='MODEL: '+str(name)+'  CURRENT EPOCH: '+str(x+1)+"/"+str(epochs)+'  BATCH SIZE: '+str(batch_size)
			for y in tqdm(range(steps_per_epoch_train), desc=epoch_desc):

				train=train_data_loader.pop()
				while train == False:
					time.sleep(10)
					train=train_data_loader.pop()
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
			acc=test_model_accuracy(model=model,transform_map=valid_transform_map,target_size=target_size,batch_size=batch_size,valid_data_loader=valid_data_loader)
			print("Model Validation Accuracy: ",acc)
			if bestModelAcc<acc:
				bestModel=deepcopy(model)
				bestModelAcc=acc
		#save best model created
		bestModel.save_weights('./weights/weights_'+name+'_'+str(round(bestModelAcc,5))+'.h5')
		bestModel.save('./models/model_'+name+'_'+str(round(bestModelAcc,5))+'.dnn')
		train_data_loader.terminate()
		valid_data_loader.terminate()
		return True
	except KeyboardInterrupt as e:
		print('Saving best model generated so far')
		bestModel.save_weights('./weights/weights_'+name+'_'+str(round(bestModelAcc,5))+'.h5')
		bestModel.save('./models/model_'+name+'_'+str(round(bestModelAcc,5))+'.dnn') 
		train_data_loader.terminate()
		valid_data_loader.terminate()
		raise KeyboardInterrupt
	except MemoryError as e:
		print('Memory Error! Saving best model generated so far')
		bestModel.save_weights('./weights/weights_'+name+'_'+str(round(bestModelAcc,5))+'.h5')
		bestModel.save('./models/model_'+name+'_'+str(round(bestModelAcc,5))+'.dnn')
		train_data_loader.terminate()
		valid_data_loader.terminate()
		return True

def test_model_accuracy(model,transform_map,target_size,batch_size,valid_data_loader):
	correct=0
	for x in tqdm(range(validDataLen),desc='Evaluating Model 		'):
		valid=valid_data_loader.pop()
		prediction=round(model.predict(valid['data'].reshape(1, target_size[0], target_size[1], 3))[0][0])
		if prediction== valid['labels'][0]:correct+=1
	return correct/validDataLen



def calBatchSize(epoch, totalEpochs):
	if epoch<=totalEpochs//6:
		return 32
	elif epoch<=(totalEpochs//6)*2:
		return 64
	elif epoch<=(totalEpochs//6)*3:
		return 128
	elif epoch<=(totalEpochs//6)*4:
		return 256
	elif epoch <=(totalEpochs//6)*5:
		return 512
	else:
		return 1024

'''
ideas for next run:
	massively increase kernal numbers, start with 128 in place of 32 and increase with the same schedule
	continue calculating receptive field size and ensure that image size does not become too much smaller than it
	try model2 again but with much larger kernal sizes, maybe drop the last conv layer as the RFS is quite a bit bigger than the image size
	impliment modelcheckpoint callback function
	increase learning rate factor to 0.4-0.7 to push past ~92% barrier

'''

#93.93
def model1():

	dropout=0.3
	kernel_size=(5,5)
	pool_size=(2,2)
	image_size=96
	epochs=50
	name='model-1'
	max_queue_size=16
	batch_size=64
	filepath='./models/model-1/'


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

	#train_model(model,epochs,name,(image_size,image_size),train_transform_map,valid_transform_map,max_queue_size)
	trainAndSaveGenerator(model,epochs,name,target_size,batch_size,filepath)

'''	(loss,acc)=model.evaluate_generator(validationGenerator(target_size,batch_size),steps=validDataLen // batch_size)

	model.save_weights('./weights/weights_'+name+'_'+str(round(acc,5))+'.h5')
	model.save('./models/model_'+name+'_'+str(round(acc,5))+'.dnn')'''
	

#added stride, removed some conv2d and dropout layers
def model2():

	dropout=0.3
	kernel_size=(5,5)
	pool_size=(2,2)
	image_size=96
	epochs=50
	name='model-2'
	max_queue_size=16
	batch_size=64
	stride=(2,2)
	filepath='./models/model-2/'



	model = Sequential()
	model.add(Conv2D(32, kernel_size=kernel_size, padding="same", strides=stride, kernel_initializer=initializers.he_normal(), input_shape=(image_size, image_size, 3)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(Dropout(dropout))
	#RFS= 1 + 2*1 = 3

	model.add(Conv2D(32, kernel_size=kernel_size, padding="same", strides=stride, kernel_initializer=initializers.he_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	#model.add(Dropout(dropout))
	#RFS = 3 + 2 * 2 = 7

	model.add(Conv2D(64, kernel_size=kernel_size, padding="same", strides=stride, kernel_initializer=initializers.he_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	#model.add(Dropout(dropout))
	#RFS = 7 + 2 * 4 = 15


	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(0.7))


	model.add(Conv2D(64, kernel_size=kernel_size, padding="same", strides=stride, kernel_initializer=initializers.he_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	#model.add(Dropout(dropout))
	#RFS = 15 + 2 * 8 = 31

	model.add(Conv2D(128, kernel_size=kernel_size, padding="same", strides=stride, kernel_initializer=initializers.he_normal()))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(Dropout(dropout))
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

	#train_model(model,epochs,name,(image_size,image_size),train_transform_map,valid_transform_map,max_queue_size)
	trainAndSaveGenerator(model,epochs,name,target_size,batch_size,filepath)

'''	(loss,acc)=model.evaluate_generator(validationGenerator(target_size,batch_size),steps=validDataLen // batch_size)

	model.save_weights('./weights/weights_'+name+'_'+str(round(acc,5))+'.h5')
	model.save('./models/model_'+name+'_'+str(round(acc,5))+'.dnn') '''

#removed a max pooling layers
def model3():

	dropout=0.3
	kernel_size=(5,5)
	pool_size=(2,2)
	image_size=96
	epochs=50
	name='model-3'
	max_queue_size=16
	batch_size=64
	filepath='./models/model-3/'


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
	#model.add(MaxPooling2D(pool_size=pool_size))
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

	#train_model(model,epochs,name,(image_size,image_size),train_transform_map,valid_transform_map,max_queue_size)
	trainAndSaveGenerator(model,epochs,name,target_size,batch_size,filepath)

'''	(loss,acc)=model.evaluate_generator(validationGenerator(target_size,batch_size),steps=validDataLen // batch_size)

	model.save_weights('./weights/weights_'+name+'_'+str(round(acc,5))+'.h5')
	model.save('./models/model_'+name+'_'+str(round(acc,5))+'.dnn') '''

#changed number of kernals (81-84)
def model4():

	dropout=0.3
	kernel_size=(5,5)
	pool_size=(2,2)
	image_size=96
	epochs=50
	name='model-4'
	batch_size=64
	filepath='./models/model-1/'


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

	#trainAndSaveBatch(model,epochs,name,(image_size,image_size))
	trainAndSaveGenerator(model,epochs,name,target_size,batch_size,filepath)

'''	(loss,acc)=model.evaluate_generator(validationGenerator(target_size,batch_size),steps=validDataLen // batch_size)

	model.save_weights('./weights/weights_'+name+'_'+str(round(acc,5))+'.h5')
	model.save('./models/model_'+name+'_'+str(round(acc,5))+'.dnn') '''



#method allows for restarting models which are trainging poorly
def modelStart(modelName):
	try:
		modelName()
		return True
	except KeyboardInterrupt as e:
		print('KeyboardInterrupt detected, ending training')
		return False


def main():
	while not modelStart(model2):
		if input('Would you like to restart this model? (y or n) ')=="n":
			break
	while not modelStart(model3):
		if input('Would you like to restart this model? (y or n) ')=="n":
			break
	while not modelStart(model4):
		if input('Would you like to restart this model? (y or n) ')=="n":
			break
	while not modelStart(model1):
		if input('Would you like to restart this model? (y or n) ')=="n":
			break
	


if __name__ == '__main__':
	main()