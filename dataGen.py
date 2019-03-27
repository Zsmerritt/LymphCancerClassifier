from skimage.io import imread
import numpy as np
import pandas as pd

class image_data_generator():

    def __init__(
         batch_size=32,
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
         adaptive_equalization=False):

        self.batch_size=batch_size
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.brightness_range = brightness_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.channel_shift_range = channel_shift_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rescale = rescale
        self.contrast_stretching = contrast_stretching
        self.adaptive_equalization = adaptive_equalization
        self.histogram_equalization = histogram_equalization


    def change_batch_size(batch_size):
        self.batch_size=batch_size

    def get_input(path):
        
        img = imread(path)
        
        return(img)


    def get_output(path,label_file=None):
        
        img_id = path.split('/')[-1].split('.')[0]
        img_id = np.int64(img_id)
        labels = label_file.loc[img_id].values
        
        return(labels)

    def process_transform(image,transformList):
        
        for 
        
        return(image)

    def get_random_transform(transformList):
        transformOutput={}
        for transform in transformList:
            transformOutput[transform]=True if np.random.uniform(0,1)>0.5 else False
        return transformOutput

        
    def image_generator(files,label_file, random):
        
        while True:
            # Select files (paths/indices) for the batch
            batch_paths = np.random.choice(a = files, 
                                             size = self.batch_size)
            batch_input = []
            batch_output = [] 
              
            # Read in each input, perform preprocessing and get labels
            for input_path in batch_paths:
                input_image = get_input(input_path )
                output_image = get_output(input_path,label_file=label_file )
                
                input_image = process_transform(image=input_image)
                batch_input += [ input_image ]
                batch_output += [ output ]
            # Return a tuple of (input,output) to feed the network
            batch_x = np.array( batch_input )
            batch_y = np.array( batch_output )
            
            yield( batch_x, batch_y )


             