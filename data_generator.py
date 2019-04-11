import numpy as np
import keras, re
import PIL.Image as pil_image
from os import walk, path

if pil_image is not None:
    _PIL_INTERPOLATION_METHODS = {
        'nearest': pil_image.NEAREST,
        'bilinear': pil_image.BILINEAR,
        'bicubic': pil_image.BICUBIC,
    }

class DataGenerator(keras.utils.Sequence):
    #'Generates data for Keras'
    #Rotation_range => how many 90 degree rotations are permitted, max 4
    def __init__(self, data_folder, batch_size=32, n_channels=3, data_format='channels_last',
                 n_classes=2, shuffle=True, rotation_range=0, vertical_flip=False, 
                 horizontal_flip=False, rescale=None, target_size=(32,32), categorical_labels=False):
        #'Initialization'
        self.batch_size = batch_size
        self.data_folder = path.abspath(data_folder)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.rotation_range=rotation_range
        self.vertical_flip=vertical_flip
        self.horizontal_flip=horizontal_flip
        self.rescale=rescale
        self.target_size=target_size
        self.categorical=categorical
        self.data_paths=self.list_pictures()
        self.labels=self.list_labels()
        self.on_epoch_end()
        if data_format == 'channels_last':
            self.row_axis=1
            self.col_axis=2
            self.channels_axis=3
        elif data_format == 'channels_first':
            self.row_axis=2
            self.col_axis=3
            self.channels_axis=1
        else:  raise ValueError(
                '`data_format` should be `"channels_last"` '
                '(channel after row and column) or '
                '`"channels_first"` (channel before row and column). '
                'Received: %s' % data_format)

    def __len__(self):
        #'Denotes the number of batches per epoch'
        return len(self.data_paths) // self.batch_size


    def __getitem__(self, index):
        #'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.data_paths[k] for k in indexes]
        list_labels_temp = [self.labels[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp, list_labels_temp)

        print(X,y)

        return X, y


    def on_epoch_end(self):
        #'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, list_IDs_temp, list_labels_temp):
        #'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.target_size, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            img=self.load_img(path=ID,target_size=self.target_size)
            # Store sample
            X[i,] = self.random_transform(img)

            # Store class
            y[i] = list_labels_temp[i]

            if self.categorical_labels: y = keras.utils.to_categorical(y, num_classes=self.n_classes)
            
        return X, y

    def get_random_transform(self):
        transform={'rotation':0,'vert_flip':0,'hor_flip':0}
        if self.rotation_range:
            transform['rotation']=np.random.randint(self.rotation_range)
        if self.vertical_flip:
            transform['vert_flip']=np.random.randint(2)
        if self.horizontal_flip:
            transform['hor_flip']=np.random.randint(2)
        return transform


    def apply_transform(self, transform, img):
        img = self.rotate_img(img, transform['rotation'])
        img_array = self.img_to_array(img)
        img_array = self.flip_axis(img_array, self.col_axis) if transform['hor_flip'] else img_array
        img_array = self.flip_axis(img_array, self.row_axis) if transform['vert_flip'] else img_array
        return img_array

    def random_transform(self, img):
        t=self.get_random_transform()
        self.apply_transform(t, img)
        return img


    def rescale_array(self, x):
        if self.rescale:
            x *= self.rescale
        return x


    def flip_axis(self, x, axis):
        x = np.asarray(x).swapaxes(axis, 0)
        x = x[::-1, ...]
        x = x.swapaxes(0, axis)
        return x


    def rotate_img(self, img, rotation):
        if rotation:
            img.rotate(rotation*90)
        return img


    def list_pictures(self, ext='jpg|jpeg|bmp|png|ppm|tif'):
        return [path.join(root, f)
                for root, _, files in walk(self.data_folder) for f in files
                if re.match(r'([\w]+\.(?:' + ext + '))', f.lower())]


    def list_labels(self):
        counter,label_dict,labels=0,{},[]
        for p in self.data_paths:
            d=path.basename(path.dirname(p))
            if d not in label_dict.keys():
                label_dict[d]=counter
                counter+=1
            labels.append(label_dict[d])
        return labels


    def load_img(self, path, grayscale=False, color_mode='rgb', target_size=None,
             interpolation='nearest'):
        if grayscale is True:
            warnings.warn('grayscale is deprecated. Please use '
                          'color_mode = "grayscale"')
            color_mode = 'grayscale'
        if pil_image is None:
            raise ImportError('Could not import PIL.Image. '
                              'The use of `array_to_img` requires PIL.')
        img = pil_image.open(path)
        if color_mode == 'grayscale':
            if img.mode != 'L':
                img = img.convert('L')
        elif color_mode == 'rgba':
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
        elif color_mode == 'rgb':
            if img.mode != 'RGB':
                img = img.convert('RGB')
        else:
            raise ValueError('color_mode must be "grayscale", "rbg", or "rgba"')
        if target_size is not None:
            width_height_tuple = (target_size[1], target_size[0])
            if img.size != width_height_tuple:
                resample = _PIL_INTERPOLATION_METHODS[interpolation]
                img = img.resize(width_height_tuple, resample)
        return img


    def img_to_array(self, img, data_format='channels_last', dtype='float32'):
        if data_format not in {'channels_first', 'channels_last'}:
            raise ValueError('Unknown data_format: %s' % data_format)
        # Numpy array x has format (height, width, channel)
        # or (channel, height, width)
        # but original PIL image has format (width, height, channel)
        x = np.asarray(img, dtype=dtype)
        if len(x.shape) == 3:
            if data_format == 'channels_first':
                x = x.transpose(2, 0, 1)
        elif len(x.shape) == 2:
            if data_format == 'channels_first':
                x = x.reshape((1, x.shape[0], x.shape[1]))
            else:
                x = x.reshape((x.shape[0], x.shape[1], 1))
        else:
            raise ValueError('Unsupported image shape: %s' % (x.shape,))
        return x

def main():
    dg=DataGenerator('./data/train/')
    print(dg.__getitem__(0))

if __name__ == '__main__':
    main()
