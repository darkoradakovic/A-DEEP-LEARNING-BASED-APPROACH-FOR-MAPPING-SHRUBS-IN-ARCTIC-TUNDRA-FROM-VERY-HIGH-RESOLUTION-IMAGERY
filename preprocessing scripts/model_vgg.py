
### VGG MODEL
# Darko Radakovic
# Montclair State University
# 15-1-25


import os
import rasterio
from rasterio.windows import Window
from skimage.transform import resize
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.models import load_model
from skimage.transform import resize
from tensorflow.keras.applications import ResNet50

from functionsdarko import (slice_image,combine_tiles_to_large_image_predictionsoneyear
                            )

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, concatenate,
                                     BatchNormalization, Activation, Dropout, Lambda)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import compute_class_weight
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt


from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization, Activation, Cropping2D
from tensorflow.keras.regularizers import l2


# slice function to create tiles from area scene
def slice_image(image_path, target_slice_size_meters=100):
    slices = []
    metadata = []
    with rasterio.open(image_path) as src:
        pixel_size = src.res[0]
        slice_size_pixels = int(target_slice_size_meters / pixel_size)

        for i in range(0, src.width, slice_size_pixels):
            for j in range(0, src.height, slice_size_pixels):
                width = min(slice_size_pixels, src.width - i)
                height = min(slice_size_pixels, src.height - j)
                window = Window(i, j, width, height)
                sliced_data = src.read(window=window)

                # Copy metadata
                slice_meta = src.meta.copy()
                slice_meta.update({
                    'height': height,
                    'width': width,
                    'transform': rasterio.windows.transform(window, src.transform)
                })

                slices.append(sliced_data)
                metadata.append(slice_meta)

    return slices, metadata


target_slice_size_meters = 200  ## in meter

# # shrubs training data local paths
# pshp_dir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train'
# p1sb_dir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/P1BS_files'
# y_train_dir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/y_train2_shrubs_p70_v2/final'
#

# # ADAPT paths
pshp_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/x_train'
p1sb_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/P1BS_files'
#y_train_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/final'
y_train_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/y_train2_shrubs_p70_v2/final'

# # wettundra
# y_train_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/y_train_wettundra3'

# # lakes and rivers
# y_train_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/y_train_lakes_and_rivers'

# VGG PRETRAINED MODEL
# vgg19_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/models/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
vgg19_path = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/models/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'




### MEMORY REDUCUCTION LOAD IMAGES

# Collect image triplets
image_triplets = []

for file_name in os.listdir(pshp_dir):
    if file_name.endswith('.tif'):
        # Full path to the PSHP and P1BS images
        pshp_image_path = os.path.join(pshp_dir, file_name)
        p1bs_image_path = pshp_image_path.replace('x_train', 'P1BS_files').replace('PSHP', 'P1BS')

        # Find y_train image
        image_path_y_train = None
        for yfilename in os.listdir(y_train_dir):
            if file_name[:-4] in yfilename and yfilename.endswith('.tif'):
                image_path_y_train = os.path.join(y_train_dir, yfilename)
                break

        # If y_train image exists, continue
        if image_path_y_train:
            image_triplets.append((pshp_image_path, p1bs_image_path, image_path_y_train))


from sklearn.model_selection import train_test_split

train_triplets, val_triplets = train_test_split(image_triplets, test_size=0.1, random_state=42)


def data_generator_from_triplets(triplets, target_slice_size_meters):
    for pshp_image_path, p1bs_image_path, image_path_y_train in triplets:
        # Slice PSHP and P1BS images
        slices_pshp, _ = slice_image(pshp_image_path, target_slice_size_meters)
        slices_p1bs, _ = slice_image(p1bs_image_path, target_slice_size_meters)
        slices_label, _ = slice_image(image_path_y_train, target_slice_size_meters)

        # each slice
        for s_pshp, s_p1bs, s_label in zip(slices_pshp, slices_p1bs, slices_label):
            # Move axis to have channels at the end
            s_pshp = np.moveaxis(s_pshp, 0, -1)  # Shape: (height, width, channels)
            s_p1bs = np.moveaxis(s_p1bs, 0, -1)  # Shape: (height, width, channels)

            # Concatenate PSHP + P1BS
            x_slice = np.concatenate((s_pshp, s_p1bs), axis=-1)  # Shape: (height, width, 5)
            x_slice = x_slice / 255.0

            # Process y_slice
            y_slice = np.moveaxis(s_label, 0, -1).astype('float32')  # Shape: (height, width, 1)

            yield x_slice, y_slice




batch_size = 8  # cnn_model3 (test wet tundra)

# Create datasets
train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator_from_triplets(train_triplets, target_slice_size_meters),
    output_signature=(
        tf.TensorSpec(shape=(400, 400, 5), dtype=tf.float32),
        tf.TensorSpec(shape=(400, 400, 1), dtype=tf.float32)
    )
)

val_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator_from_triplets(val_triplets, target_slice_size_meters),
    output_signature=(
        tf.TensorSpec(shape=(400, 400, 5), dtype=tf.float32),
        tf.TensorSpec(shape=(400, 400, 1), dtype=tf.float32)
    )
)


train_dataset = train_dataset.shuffle(buffer_size=1000)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

val_dataset = val_dataset.batch(batch_size)
val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)




### --- --- --- VGG --- --- ---

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, UpSampling2D, Concatenate,
                                     BatchNormalization, Activation, Cropping2D)
from tensorflow.keras.applications import VGG19  # or VGG16




def build_vgg_unet_5channels(
    vgg_weights_path: str,
    input_shape=(400,400,5)
):
    """
    VGG19 encoder + U-Net decoder. input shape (400,400,5),
    but only 3 channels extracted

    Steps:
      1) Input of shape (400,400,5).
      2) first 3 channels extracted -> shape (400,400,3).
      3) Output is a 1-channel segmentation mask.
    """

    ### (A) input: (400,400,5)
    inputs_5ch = Input(shape=input_shape, name="input_5ch")

    ### (B) Slice out 3 channels
    three_ch = Lambda(
        lambda x: x[:,:,:,0:3],
        name="slice_3ch"
    )(inputs_5ch)

    ### (C) Create a plain VGG19 (no top), with shape = (None, None, 3).
    #  no input_tensor here. skip outputs manually.
    vgg_base = VGG19(
        weights=None,
        include_top=False,
        input_shape=(None, None, 3)
    )
    # Load the pretrained weights from .h5
    vgg_base.load_weights(vgg_weights_path)

    ### Identify skip layers by name:
    #  block1_conv2 => ~ (None, H/2,  W/2,   64)
    #  block2_conv2 => ~ (None, H/4,  W/4,  128)
    #  block3_conv3 => ~ (None, H/8,  W/8,  256)
    #  block4_conv3 => ~ (None, H/16, W/16, 512)
    #  block5_conv3 => ~ (None, H/32, W/32, 512)
    s1 = vgg_base.get_layer('block1_conv2').output
    s2 = vgg_base.get_layer('block2_conv2').output
    s3 = vgg_base.get_layer('block3_conv3').output
    s4 = vgg_base.get_layer('block4_conv3').output
    encoder_out = vgg_base.get_layer('block5_conv3').output

    ### encoder
    encoder_sub = Model(
        inputs=vgg_base.input,   # shape(None,None,3)
        outputs=[s1, s2, s3, s4, encoder_out],
        name='vgg19_encoder_sub'
    )

    ###  3-ch tensor
    skip1, skip2, skip3, skip4, enc_out = encoder_sub(three_ch)

    ### (D) U-Net decoder
    # enc_out shape ~ (None,H/32,W/32,512)
    x = enc_out

    # Step 1 => skip4
    x = UpSampling2D((2,2))(x)  # H/32 -> H/16
    x = Conv2D(512, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Concatenate()([x, skip4])

    # Step 2 => skip3
    x = UpSampling2D((2,2))(x)  # H/16->H/8
    x = Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Concatenate()([x, skip3])

    # Step 3 => skip2
    x = UpSampling2D((2,2))(x)  # H/8->H/4
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Concatenate()([x, skip2])

    # Step 4 => skip1
    x = UpSampling2D((2,2))(x)  # H/4->H/2
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Concatenate()([x, skip1])

    # # final upsample from H/2 -> H
    # x = UpSampling2D((2,2))(x)
    # x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    # x = BatchNormalization()(x)

    # final mask => 1 channel, sigmoid
    outputs = Conv2D(1, (1,1), padding='same', activation='sigmoid')(x)

    ### output:
    #   input = (400,400,5), output = (400,400,1)
    model_5ch = Model(inputs=inputs_5ch, outputs=outputs, name="vgg19_unet_5ch")
    return model_5ch



import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# vgg_unet_5ch = build_vgg_unet_5channels(input_shape=(400,400,5))
# vgg_unet_5ch = build_vgg_unet(input_shape=(256,256,3))
model = build_vgg_unet_5channels(
        vgg_weights_path=vgg19_path,
        input_shape=(400,400,5)
    )


model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])

# Create callbacks
checkpoint = ModelCheckpoint('shrub_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT.tf',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)

# checkpoint = ModelCheckpoint('shrub_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v1_ADAPT.tf', monitor='val_loss', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)



# Train
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=100,
    callbacks=[early_stopping, checkpoint]
)





