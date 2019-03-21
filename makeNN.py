from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import initializers
from copy import deepcopy


# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        fill_mode='nearest')


# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

trainDataLen=128909+87757
validDataLen=2000+1361

trainDataLenP=2000
validDataLenP=200

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

def trainAndSave(model,epochs,name,image_size,trainDataLen,validDataLen):
	#hold on to best model to save after training
	bestModel=model
	bestModelLoss,bestModelAcc=1.0,0.0

	#moved these out of loop to solve mem alocation prob
	initBatchSize=16
	trainGen=trainGenerator(image_size,initBatchSize)
	validGen=validationGenerator(image_size,initBatchSize)

	for x in range(1,epochs+1):
		#print infor and adjust batch size
		print('MODEL: ',name,' CURRENT EPOCH:',x)
		batch_size=calBatchSize(x,epochs)
		#update generators only when batch size changes
		if batch_size!=initBatchSize:
			trainGen=trainGenerator(image_size,batch_size)
			validGen=validationGenerator(image_size,batch_size)
		#fit model
		model.fit_generator(
		        trainGen,
		        #steps_per_epoch=trainDataLen // batch_size,
		        steps_per_epoch=trainDataLenP // batch_size,
		        epochs=1,
		        validation_data=validationGenerator(image_size,batch_size),
		        #validation_steps=validDataLen // batch_size,
		        validation_steps=validDataLenP // 16,
		        verbose=1,
		        max_queue_size=16)
		#cal loss and accuracy before comparing to previous best model
		loss,acc=model.evaluate_generator(validGen)
		if bestModelAcc<acc and bestModelLoss>loss:
			bestModel=deepcopy(model)
			bestModelLoss,bestModelAcc=loss,acc
	#save best model created
	bestModel.save_weights('./weights/weights_'+name+'_'+str(round(bestModelAcc,5))+'.h5')
	bestModel.save('./models/model_'+name+'_'+str(round(bestModelAcc,5))+'.dnn') 


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

#~86% max
def model1():

	dropout=0.3
	kernel_size=(3,3)
	pool_size=(2,2)
	image_size=150
	epochs=50
	name='prototype1-og'

	model = Sequential()
	model.add(Conv2D(32, kernel_size=kernel_size, padding='same', kernel_initializer=initializers.he_normal(seed=None), input_shape=(image_size, image_size, 3)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(32, kernel_size=kernel_size, padding='same', kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(64, kernel_size=kernel_size, padding='same', kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(64, kernel_size=kernel_size, padding='same', kernel_initializer=initializers.he_normal(seed=None)))
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
	              optimizer='rmsprop',
	              metrics=['accuracy'])

	trainAndSave(model,epochs,name,image_size)

#added 256 dense layer
def model2():

	dropout=0.3
	kernel_size=(3,3)
	pool_size=(2,2)
	image_size=150
	epochs=50
	name='prototype2'

	model = Sequential()
	model.add(Conv2D(32, kernel_size=kernel_size, padding='same', kernel_initializer=initializers.he_normal(seed=None), input_shape=(image_size, image_size, 3)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(32, kernel_size=kernel_size, padding='same', kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(64, kernel_size=kernel_size, padding='same', kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(64, kernel_size=kernel_size, padding='same', kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))


	model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

	model.add(Dense(256, kernel_initializer=initializers.lecun_normal(seed=None)))
	model.add(Activation('relu'))
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
	              optimizer='rmsprop',
	              metrics=['accuracy'])

	trainAndSave(model,epochs,name,image_size)

#added 128 and 256 conv layers
def model3():

	dropout=0.3
	kernel_size=(3,3)
	pool_size=(2,2)
	image_size=150
	epochs=50
	name='prototype3'

	model = Sequential()
	model.add(Conv2D(32, kernel_size=kernel_size, padding='same', kernel_initializer=initializers.he_normal(seed=None), input_shape=(image_size, image_size, 3)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(32, kernel_size=kernel_size, padding='same', kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(64, kernel_size=kernel_size, padding='same', kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(64, kernel_size=kernel_size, padding='same', kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(128, kernel_size=kernel_size, padding='same', kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(256, kernel_size=kernel_size, padding='same', kernel_initializer=initializers.he_normal(seed=None)))
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
	              optimizer='rmsprop',
	              metrics=['accuracy'])

	trainAndSave(model,epochs,name,image_size)

#increased dropout to .5
def model4():

	dropout=0.5
	kernel_size=(3,3)
	pool_size=(2,2)
	image_size=150
	epochs=50
	name='prototype4'

	model = Sequential()
	model.add(Conv2D(32, kernel_size=kernel_size, padding='same', kernel_initializer=initializers.he_normal(seed=None), input_shape=(image_size, image_size, 3)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(32, kernel_size=kernel_size, padding='same', kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(64, kernel_size=kernel_size, padding='same', kernel_initializer=initializers.he_normal(seed=None)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(dropout))

	model.add(Conv2D(64, kernel_size=kernel_size, padding='same', kernel_initializer=initializers.he_normal(seed=None)))
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
	              optimizer='rmsprop',
	              metrics=['accuracy'])

	trainAndSave(model,epochs,name,image_size)

def main():
	model1()
	model2()
	model3()
	model4()



if __name__ == '__main__':
	main()