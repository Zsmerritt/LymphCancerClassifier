"""Utilities for real-time data augmentation on image data.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np, pandas as pd, glob, os
from keras.preprocessing.image  import *

'''
(random_rotation,
random_shift,
random_shear,
random_zoom,
apply_brightness_shift,
random_brightness,
transform_matrix_offset_center,
apply_affine_transform,
apply_channel_shift,
random_channel_shift,
flip_axis,
array_to_img,
img_to_array,
save_img,
load_img,
list_pictures)
'''

from skimage.color.adapt_rgb import adapt_rgb, each_channel

import numpy as np
import re
from six.moves import range
import os
import threading
import warnings
import multiprocessing.pool
from keras_preprocessing import get_keras_submodule

try:
	IteratorType = get_keras_submodule('utils').Sequence
except ImportError:
	IteratorType = object

try:
	from PIL import ImageEnhance
	from PIL import Image as pil_image
except ImportError:
	pil_image = None
	ImageEnhance = None

try:
	import scipy
	# scipy.linalg cannot be accessed until explicitly imported
	from scipy import linalg
	# scipy.ndimage cannot be accessed until explicitly imported
	from scipy import ndimage
except ImportError:
	scipy = None

from skimage import data, img_as_float
from skimage import exposure

if pil_image is not None:
	_PIL_INTERPOLATION_METHODS = {
		'nearest': pil_image.NEAREST,
		'bilinear': pil_image.BILINEAR,
		'bicubic': pil_image.BICUBIC,
	}
	# These methods were only introduced in version 3.4.0 (2016).
	if hasattr(pil_image, 'HAMMING'):
		_PIL_INTERPOLATION_METHODS['hamming'] = pil_image.HAMMING
	if hasattr(pil_image, 'BOX'):
		_PIL_INTERPOLATION_METHODS['box'] = pil_image.BOX
	# This method is new in version 1.1.3 (2013).
	if hasattr(pil_image, 'LANCZOS'):
		_PIL_INTERPOLATION_METHODS['lanczos'] = pil_image.LANCZOS



def get_transform_map(
	 data_folder,
	 rotation_range=0,
	 width_shift_range=0.,
	 height_shift_range=0.,
	 brightness_range=None,
	 shear_range=0.,
	 zoom_range=0.,
	 channel_shift_range=0.,
	 horizontal_flip=False,
	 vertical_flip=False,
	 rescale=None,
	 contrast_stretching=False, 
	 histogram_equalization=False, 
	 adaptive_equalization=False,
	 seed=None,
	 featurewise_center=False,
	 samplewise_center=False,
	 featurewise_std_normalization=False,
	 samplewise_std_normalization=False,
	 zca_whitening=False,
	 zca_epsilon=1e-6,
	 fill_mode='nearest',
	 cval=0.,
	 preprocessing_function=None,
	 data_format='channels_last',
	 validation_split=0.0,
	 dtype='float32'):
	
	transform_map={
	'data_folder' : data_folder,
	'rotation_range' : rotation_range,
	'width_shift_range' : width_shift_range,
	'height_shift_range' : height_shift_range,
	'brightness_range' : brightness_range,
	'shear_range' : shear_range,
	'zoom_range' : zoom_range,
	'channel_shift_range' : channel_shift_range,
	'horizontal_flip' : horizontal_flip,
	'vertical_flip' : vertical_flip,
	'rescale' : rescale,
	'contrast_stretching' : contrast_stretching,
	'adaptive_equalization' : adaptive_equalization,
	'histogram_equalization' : histogram_equalization,
	'seed' : seed,
	'featurewise_center' : featurewise_center,
	'samplewise_center' : samplewise_center,
	'featurewise_std_normalization' : featurewise_std_normalization,
	'samplewise_std_normalization' : samplewise_std_normalization,
	'zca_whitening' : zca_whitening,
	'zca_epsilon' : zca_epsilon,
	'fill_mode' : fill_mode,
	'cval' : cval,
	'preprocessing_function' : preprocessing_function,
	'dtype' : dtype,
	}


	if data_format not in {'channels_last', 'channels_first'}:
		raise ValueError(
			'`data_format` should be `"channels_last"` '
			'(channel after row and column) or '
			'`"channels_first"` (channel before row and column). '
			'Received: %s' % data_format)
	transform_map['data_format'] = data_format
	if data_format == 'channels_first':
		transform_map['channel_axis'] = 1
		transform_map['row_axis'] = 2
		transform_map['col_axis'] = 3
	if data_format == 'channels_last':
		transform_map['channel_axis'] = 3
		transform_map['row_axis'] = 1
		transform_map['col_axis'] = 2
	if validation_split and not 0 < validation_split < 1:
		raise ValueError(
			'`validation_split` must be strictly between 0 and 1. '
			' Received: %s' % validation_split)
	transform_map['_validation_split'] = validation_split

	transform_map['mean'] = None
	transform_map['std'] = None
	transform_map['principal_components'] = None

	if np.isscalar(zoom_range):
		transform_map['zoom_range'] = [1 - zoom_range, 1 + zoom_range]
	elif len(zoom_range) == 2:
		transform_map['zoom_range'] = [zoom_range[0], zoom_range[1]]
	else:
		raise ValueError('`zoom_range` should be a float or '
						 'a tuple or list of two floats. '
						 'Received: %s' % (zoom_range,))
	if zca_whitening:
		if not featurewise_center:
			transform_map['featurewise_center'] = True
			warnings.warn('This ImageDataGenerator specifies '
						  '`zca_whitening`, which overrides '
						  'setting of `featurewise_center`.')
		if featurewise_std_normalization:
			transform_map['featurewise_std_normalization'] = False
			warnings.warn('This ImageDataGenerator specifies '
						  '`zca_whitening` '
						  'which overrides setting of'
						  '`featurewise_std_normalization`.')
	if featurewise_std_normalization:
		if not featurewise_center:
			transform_map['featurewise_center'] = True
			warnings.warn('This ImageDataGenerator specifies '
						  '`featurewise_std_normalization`, '
						  'which overrides setting of '
						  '`featurewise_center`.')
	if samplewise_std_normalization:
		if not samplewise_center:
			transform_map['samplewise_center'] = True
			warnings.warn('This ImageDataGenerator specifies '
						  '`samplewise_std_normalization`, '
						  'which overrides setting of '
						  '`samplewise_center`.')

	return transform_map


	
def image_generator(transform_map, batch_size, target_size):
	image_paths = glob.glob(pathname=transform_map['data_folder']+'/**/*.tif', recursive=True)
	while True:
		# Select files (paths/indices) for the batch
		batch_paths = np.random.choice(a = image_paths, 
										 size = batch_size)
		batch_input = []
		batch_output = [] 
		  
		# Read in each input, perform preprocessing and get labels
		for image_path in batch_paths:
			#open and convert image to array before closing
			image=pil_image.open(image_path)

			#resize image to correct size
			image.thumbnail(target_size)

			trans_image=random_transform(np.asarray(image),transform_map)
			image.close()
			#get label of image
			output=image_path.split('/')[-2]
			output = 0 if output[0]=="n" else 1

			batch_input += [ trans_image ]
			batch_output += [ output ]
		# Return a tuple of (input,output) to feed the network
		batch_x = np.array( batch_input )
		batch_y = np.array( batch_output )
		
		yield( batch_x, batch_y )


def image_processor(transform_map, target_size,image_multiplier=1,save_test_images=False,save_test_directory='./data/testImages'):
	image_paths = glob.glob(pathname=transform_map['data_folder']+'/**/*.tif', recursive=True)
	# Select files (paths/indices) for the batch
	batch_input = []
	batch_output = [] 
	  
	# Read in each input, perform preprocessing and get labels
	for image_path in image_paths:
		image=pil_image.open(image_path)
		image.thumbnail(target_size)
		output=image_path.split('/')[-2]
		output = 0 if output[0]=="n" else 1
		for x in range(image_multiplier):

			trans_image=random_transform(np.asarray(image),transform_map)

			batch_input += [ trans_image ]
			batch_output += [ output ]

			if save_test_images and (x==1 or x==2 or x==3):
				image_name=output=image_path.split('/')[-1].split('.')[0]
				(array_to_img(trans_image)).save(save_test_directory+image_name+'--'+str(x), 'JPEG')

		image.close()
	# Return a tuple of (input,output) to feed the network
	batch_x = np.array( batch_input )
	batch_y = np.array( batch_output )
	
	return {'data':batch_x, 'labels':batch_y}



def get_random_transform(transform_map, img_shape, seed=None):
	"""Generates random parameters for a transformation.

	# Arguments
		seed: Random seed.
		img_shape: Tuple of integers.
			Shape of the image that is transformed.

	# Returns
		A dictionary containing randomly chosen parameters describing the
		transformation.
	"""
	img_row_axis = transform_map['row_axis'] - 1
	img_col_axis = transform_map['col_axis'] - 1

	if seed is not None:
		np.random.seed(seed)

	if transform_map['rotation_range']:
		theta = np.random.uniform(
			-transform_map['rotation_range'],
			transform_map['rotation_range'])
	else:
		theta = 0

	if transform_map['height_shift_range']:
		try:  # 1-D array-like or int
			tx = np.random.choice(transform_map['height_shift_range'])
			tx *= np.random.choice([-1, 1])
		except ValueError:  # floating point
			tx = np.random.uniform(-transform_map['height_shift_range'],
								   transform_map['height_shift_range'])
		if np.max(transform_map['height_shift_range']) < 1:
			tx *= img_shape[img_row_axis]
	else:
		tx = 0

	if transform_map['width_shift_range']:
		try:  # 1-D array-like or int
			ty = np.random.choice(transform_map['width_shift_range'])
			ty *= np.random.choice([-1, 1])
		except ValueError:  # floating point
			ty = np.random.uniform(-transform_map['width_shift_range'],
								   transform_map['width_shift_range'])
		if np.max(transform_map['width_shift_range']) < 1:
			ty *= img_shape[img_col_axis]
	else:
		ty = 0

	if transform_map['shear_range']:
		shear = np.random.uniform(
			-transform_map['shear_range'],
			transform_map['shear_range'])
	else:
		shear = 0

	if transform_map['zoom_range'][0] == 1 and transform_map['zoom_range'][1] == 1:
		zx, zy = 1, 1
	else:
		zx, zy = np.random.uniform(
			transform_map['zoom_range'][0],
			transform_map['zoom_range'][1],
			2)

	flip_horizontal = (np.random.random() < 0.5) * transform_map['horizontal_flip']
	flip_vertical = (np.random.random() < 0.5) * transform_map['vertical_flip']

	channel_shift_intensity = None
	if transform_map['channel_shift_range'] != 0:
		channel_shift_intensity = np.random.uniform(-transform_map['channel_shift_range'],
													transform_map['channel_shift_range'])

	brightness = None
	if transform_map['brightness_range'] is not None:
		if len(transform_map['brightness_range']) != 2:
			raise ValueError(
				'`brightness_range should be tuple or list of two floats. '
				'Received: %s' % (transform_map['brightness_range'],))
		brightness = np.random.uniform(transform_map['brightness_range'][0],
									   transform_map['brightness_range'][1])

	contrast_stretching=False
	if transform_map['contrast_stretching']:
		if np.random.random() < 0.5:
			contrast_stretching=True
	
	adaptive_equalization=False
	if transform_map['adaptive_equalization']:
		if np.random.random() < 0.5:
			adaptive_equalization=True
	
	histogram_equalization=False
	if transform_map['histogram_equalization']:
		if np.random.random() < 0.5:
			histogram_equalization=True

	transform_parameters = {'theta': theta,
							'tx': tx,
							'ty': ty,
							'shear': shear,
							'zx': zx,
							'zy': zy,
							'flip_horizontal': flip_horizontal,
							'flip_vertical': flip_vertical,
							'channel_shift_intensity': channel_shift_intensity,
							'brightness': brightness,
							'contrast_stretching':contrast_stretching,
							'adaptive_equalization':adaptive_equalization,
							'histogram_equalization':histogram_equalization
							}

	return transform_parameters


def apply_transform(x, transform_parameters, transform_map):
	"""Applies a transformation to an image according to given parameters.

	# Arguments
		x: 3D tensor, single image.
		transform_parameters: Dictionary with string - parameter pairs
			describing the transformation.
			Currently, the following parameters
			from the dictionary are used:
			- `'theta'`: Float. Rotation angle in degrees.
			- `'tx'`: Float. Shift in the x direction.
			- `'ty'`: Float. Shift in the y direction.
			- `'shear'`: Float. Shear angle in degrees.
			- `'zx'`: Float. Zoom in the x direction.
			- `'zy'`: Float. Zoom in the y direction.
			- `'flip_horizontal'`: Boolean. Horizontal flip.
			- `'flip_vertical'`: Boolean. Vertical flip.
			- `'channel_shift_intencity'`: Float. Channel shift intensity.
			- `'brightness'`: Float. Brightness shift intensity.

	# Returns
		A transformed version of the input (same shape).
	"""
	# x is a single image, so it doesn't have image number at index 0
	img_row_axis = transform_map['row_axis'] - 1
	img_col_axis = transform_map['col_axis'] - 1
	img_channel_axis = transform_map['channel_axis'] - 1

	x = apply_affine_transform(x, transform_parameters.get('theta', 0),
							   transform_parameters.get('tx', 0),
							   transform_parameters.get('ty', 0),
							   transform_parameters.get('shear', 0),
							   transform_parameters.get('zx', 1),
							   transform_parameters.get('zy', 1),
							   row_axis=img_row_axis,
							   col_axis=img_col_axis,
							   channel_axis=img_channel_axis,
							   fill_mode=transform_map['fill_mode'],
							   cval=transform_map['cval'])

	if transform_parameters.get('channel_shift_intensity') is not None:
		x = apply_channel_shift(x,
								transform_parameters['channel_shift_intensity'],
								img_channel_axis)

	if transform_parameters.get('flip_horizontal', False):
		x = flip_axis(x, img_col_axis)

	if transform_parameters.get('flip_vertical', False):
		x = flip_axis(x, img_row_axis)

	if transform_parameters.get('contrast_stretching'):
		p2, p98 = np.percentile(x, (2, 98))
		x = exposure.rescale_intensity(x, in_range=(p2, p98))

	if transform_parameters.get('adaptive_equalization'):
		x = exposure.equalize_adapthist(x, clip_limit=0.03)

	if transform_parameters.get('histogram_equalization'):
		x = exposure.equalize_hist(x)

	if transform_parameters.get('brightness') is not None:
		x = apply_brightness_shift(x, transform_parameters['brightness'])
	


	return x

def random_transform(image, transform_map, seed=None):
	"""Applies a random transformation to an image.

	# Arguments
		x: 3D tensor, single image.
		seed: Random seed.

	# Returns
		A randomly transformed version of the input (same shape).
	"""
	params = get_random_transform(transform_map, image.shape, seed)
	return apply_transform(image, params, transform_map)


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x
