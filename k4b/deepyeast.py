import csv
import numpy as np
import os
import math

import itertools
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from skimage.io import imread
from skimage.transform import resize

import keras
import keras.backend as K
from keras.utils import np_utils
from keras.utils.data_utils import get_file
from keras_applications.imagenet_utils import _obtain_input_shape
from keras import layers,models
from keras.regularizers import l2


def load_label_names(mode="main"):
    ''''
    returns list of label names used in the study. mode="main" by default
    switch to true to get labels for transfer learning dataset
    '''
    if mode == "main":
        label_names = ['Cell_periphery','Cytoplasm',
               'endosome','ER','Golgi',
               'Mitochondrion','Nuclear_Periphery',
               'Nucleolus','Nucleus','Peroxisome',
               'Spindle_pole','Vacuole']
    elif mode == "transfer":
        label_names = ['actin','bud neck','lipid practice','microtubule']
    else:
        print("mode must be main or transfer.")
        return 0

    return label_names

def number2label(label,mode="main"):
    '''
    input: a yeast subcellular localization number(s) of int (or numpy with length 1)
    returns:decoded localization string(s) numpy
    transfer=False by default for training dataset
    switch to True to get labels for transfer learning dataset
    '''
    label_names=load_label_names(mode=mode)
    return label_names[label]

def load_data(mode="main"):
    '''
    Codes adopted from keras https://github.com/keras-team/keras/blob/master/keras/utils/data_utils.py
    Downloads and load files from if it not already in the cache..
    by default, it will be saved to ~/.keras

    mode = "main"(default) or "transfer"
    '''
    print("Will load train,test,valid data for dataset: %s" % mode)
    if mode == "main":
        paths = ["main.tar.gz","HOwt_train.txt","HOwt_val.txt","HOwt_test.txt"]
        data_path = get_file(paths[0],
                             origin="http://kodu.ut.ee/~leopoldp/2016_DeepYeast/data/main.tar.gz",
                             extract=True,
                             cache_subdir='deepyeast')
        train_path = get_file(paths[1],origin="http://kodu.ut.ee/~leopoldp/2016_DeepYeast/code/reports/HOwt_train.txt",
                             cache_subdir='deepyeast')
        val_path = get_file(paths[2],origin="http://kodu.ut.ee/~leopoldp/2016_DeepYeast/code/reports/HOwt_val.txt",
                             cache_subdir='deepyeast')
        test_path = get_file(paths[3],origin="http://kodu.ut.ee/~leopoldp/2016_DeepYeast/code/reports/HOwt_test.txt",
                             cache_subdir='deepyeast')
    elif mode == "transfer":
        paths = ["transfer.tar.gz","HOwt_transfer_train.txt","HOwt_transfer_val.txt","HOwt_transfer_test.txt"]

        data_path = get_file(paths[0], 
                             origin='http://kodu.ut.ee/~leopoldp/2016_DeepYeast/data/transfer.tar.gz',
                             extract=True,
                             cache_subdir='deepyeast_transfer')   
        train_path = get_file(paths[1], 
                              origin='http://kodu.ut.ee/~leopoldp/2016_DeepYeast/code/image_prep/data/HOwt_transfer_train.txt', 
                              cache_subdir='deepyeast_transfer')
        val_path = get_file(paths[2], 
                            origin='http://kodu.ut.ee/~leopoldp/2016_DeepYeast/code/image_prep/data/HOwt_transfer_val.txt', 
                            cache_subdir='deepyeast_transfer')
        test_path = get_file(paths[3], 
                             origin='http://kodu.ut.ee/~leopoldp/2016_DeepYeast/code/image_prep/data/HOwt_transfer_test.txt', 
                             cache_subdir='deepyeast_transfer')
    else:
        print("mode must be main or transfer.")
        return 0
    
    data_path, _ = os.path.split(data_path)

    X_train = []
    X_valid = []
    X_test = []
    y_train = []
    y_valid = []
    y_test = []

    with open(train_path) as f:
        count = sum(1 for row in csv.reader(f))
        print("loading train dataset with %s images" % count)
        f.seek(0)
        reader = csv.reader(f)    
        
        for row in reader: 
            row = row[0].split(" ") 
            image_path = row[0]
            image_path = os.path.join(data_path,image_path)
            image = imread(image_path)
            image = resize(image,(64,64)) #this will change the numpy range to 0 to 1
            X_train.append(image)    
            label = row[1]
            y_train.append(label)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    y_train = np_utils.to_categorical(y_train)

    with open(val_path) as f:
        count = sum(1 for row in csv.reader(f))
        print("loading valid dataset with %s images" % count)
        f.seek(0)
        reader = csv.reader(f)
        for row in reader: 
            row = row[0].split(" ") 
            image_path = row[0]
            image_path = os.path.join(data_path,image_path)
            image = imread(image_path)
            image = resize(image,(64,64))
            X_valid.append(image)
            label = row[1]
            y_valid.append(label)
    X_valid = np.array(X_valid)
    y_valid = np.array(y_valid)
    y_valid = np_utils.to_categorical(y_valid)

    with open(test_path) as f:
        count = sum(1 for row in csv.reader(f))
        print("loading test dataset with %s images" % count)
        f.seek(0)
        reader = csv.reader(f)
        for row in reader:
            row = row[0].split(" ")
            image_path = row[0]
            image_path = os.path.join(data_path,image_path)
            image = imread(image_path)
            image = resize(image,(64,64)) 
            X_test.append(image)
            label = row[1]
            y_test.append(label)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    y_test = np_utils.to_categorical(y_test)

    print("train input shape: %s" % str(X_train.shape))
    print("train label shape: %s" % str(y_train.shape))
    print("validation input shape: %s" % str(X_valid.shape))
    print("validation label shape: %s" % str(y_valid.shape))
    print("test input shape: %s" % str(X_test.shape))
    print("test label shape: %s" % str(y_test.shape))

    return X_train, y_train, X_valid, y_valid,X_test,y_test

def DeepYeast_model(include_top=True,
          weights=None,
          input_tensor=None,
          input_shape=None,
          pooling=None,
          classes=12,
          **kwargs):
    """Instantiates the DEEPYEAST architecture.
    #based on keras vgg16 model@github.
    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)`
            (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    #backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)
    if not (weights in {'pretrained', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `pretrained` '
                         '(pre-training on deepyeast train dataset), '
                         'or the path to the weights file to be loaded.')

    if weights == 'pretrained' and include_top and classes != 12:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 12')
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=64,
                                      min_size=1,
                                      data_format='channels_last',
                                      require_flatten=include_top,
                                      weights=weights)
    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Block 1
    x = layers.Conv2D(64, (3, 3),
                      padding='same',
                      name='block1_conv1',kernel_initializer="glorot_normal")(img_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, (3, 3),
                      padding='same',
                      name='block1_conv2',kernel_initializer="glorot_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3),
                      padding='same',
                      name='block2_conv1',kernel_initializer="glorot_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, (3, 3),
                      padding='same',
                      name='block2_conv2',kernel_initializer="glorot_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      padding='same',
                      name='block3_conv1',kernel_initializer="glorot_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(256, (3, 3),
                      padding='same',
                      name='block3_conv2',kernel_initializer="glorot_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(256, (3, 3),
                      padding='same',
                      name='block3_conv3',kernel_initializer="glorot_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(256, (3, 3),
                      padding='same',
                      name='block3_conv4',kernel_initializer="glorot_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    if include_top:
        # Classification block
        x = layers.Flatten(name='flatten')(x)
        x = layers.Dense(512, name='fc1',
                         kernel_initializer="glorot_normal",
                        kernel_regularizer=l2(0.0005))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512,name='fc2',
                         kernel_initializer="glorot_normal",
                        kernel_regularizer=l2(0.0005))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
        
    # Create model.
    model = models.Model(inputs, x, name='deepyeast')
    sgd = keras.optimizers.sgd(lr=0.1)
    model.compile(sgd,loss="categorical_crossentropy",metrics=["acc"])
    # Load weights.
    '''
    if weights == 'imagenet':
        if include_top:
            weights_path = keras_utils.get_file(
                'vgg19_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                file_hash='cbe5617147190e668d6c5d5026f83318')
        else:
            weights_path = keras_utils.get_file(
                'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                file_hash='253f8cb515780f3b799900260a226db6')
        model.load_weights(weights_path)
        if backend.backend() == 'theano':
            keras_utils.convert_all_kernels_in_model(model)
    elif weights is not None:
        model.load_weights(weights)
    '''
    return model

def step_decay(epoch):
    #https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 25.0
    lrate = initial_lrate * math.pow(drop,  
           math.floor((1+epoch)/epochs_drop))
    print("current learning rate:%e" % lrate)
    return lrate
