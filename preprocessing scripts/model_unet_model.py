


### UNET MODEL ENHANCED 16-1-25

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



def slice_image(image_path, target_slice_size_meters=100):
    slices = []
    metadata = []  # List to store metadata for each slice
    with rasterio.open(image_path) as src:
        pixel_size = src.res[0]
        slice_size_pixels = int(target_slice_size_meters / pixel_size)

        for i in range(0, src.width, slice_size_pixels):
            for j in range(0, src.height, slice_size_pixels):
                width = min(slice_size_pixels, src.width - i)
                height = min(slice_size_pixels, src.height - j)
                window = Window(i, j, width, height)
                sliced_data = src.read(window=window)

                # Copy metadata and update relevant fields for the slice
                slice_meta = src.meta.copy()
                slice_meta.update({
                    'height': height,
                    'width': width,
                    'transform': rasterio.windows.transform(window, src.transform)
                })

                slices.append(sliced_data)
                metadata.append(slice_meta)  # Store metadata for the slice

    return slices, metadata


target_slice_size_meters = 200  ## in meter

# # shrubs
# pshp_dir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train'
# p1sb_dir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/P1BS_files'
# y_train_dir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/y_train2_shrubs_p70_v2/final'


# # ADAPT
pshp_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/x_train'
p1sb_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/P1BS_files'
#y_train_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/final'
y_train_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/y_train2_shrubs_p70_v2/final'

# # wettundra
# y_train_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/y_train_wettundra3'

# # lakes and rivers
# y_train_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/y_train_lakes_and_rivers'



### MEMORY REDUCUCTION LOAD IMAGES

# Collect image triplets
image_triplets = []

for file_name in os.listdir(pshp_dir):
    if file_name.endswith('.tif'):
        # Full path to the PSHP and P1BS images
        pshp_image_path = os.path.join(pshp_dir, file_name)
        p1bs_image_path = pshp_image_path.replace('x_train', 'P1BS_files').replace('PSHP', 'P1BS')

        # Find corresponding y_train image
        image_path_y_train = None
        for yfilename in os.listdir(y_train_dir):
            if file_name[:-4] in yfilename and yfilename.endswith('.tif'):
                image_path_y_train = os.path.join(y_train_dir, yfilename)
                break  # Stop searching once the correct label is found

        # If corresponding y_train image exists, proceed
        if image_path_y_train:
            image_triplets.append((pshp_image_path, p1bs_image_path, image_path_y_train))


from sklearn.model_selection import train_test_split

train_triplets, val_triplets = train_test_split(image_triplets, test_size=0.1, random_state=42)




# Define the data generator function
def data_generator_from_triplets(triplets, target_slice_size_meters):
    for pshp_image_path, p1bs_image_path, image_path_y_train in triplets:
        # Slice PSHP and P1BS images
        slices_pshp, _ = slice_image(pshp_image_path, target_slice_size_meters)
        slices_p1bs, _ = slice_image(p1bs_image_path, target_slice_size_meters)
        slices_label, _ = slice_image(image_path_y_train, target_slice_size_meters)

        # For each slice, process and yield
        for s_pshp, s_p1bs, s_label in zip(slices_pshp, slices_p1bs, slices_label):
            # Move axis to have channels at the end
            s_pshp = np.moveaxis(s_pshp, 0, -1)  # Shape: (height, width, channels)
            s_p1bs = np.moveaxis(s_p1bs, 0, -1)  # Shape: (height, width, channels)

            # Concatenate PSHP and P1BS along the channel axis to form x_slice
            x_slice = np.concatenate((s_pshp, s_p1bs), axis=-1)  # Shape: (height, width, 5)
            x_slice = x_slice / 255.0

            # Process y_slice
            y_slice = np.moveaxis(s_label, 0, -1).astype('float32')  # Shape: (height, width, 1)

            yield x_slice, y_slice




batch_size = 8  # cnn_model3 (test wet tundra)

# Create datasets with output_signature
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


# Configure datasets
train_dataset = train_dataset.shuffle(buffer_size=1000)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

val_dataset = val_dataset.batch(batch_size)
val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)





## --- --- UNET MODELS --- ---

from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization)
from tensorflow.keras.models import Model

def unet_model_256(height, width, channels):
    """
    A U-Net model that reaches up to 256 filters in the bottleneck layer.
    It does three downsampling steps (64->128->256), then the bottleneck (256),
    and three corresponding upsampling steps.
    """

    inputs = Input((height, width, channels))

    # ----- ENCODER -----
    # (1) Downsampling block 1
    c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2))(c1)  # -> half spatial size

    # (2) Downsampling block 2
    c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2))(c2)  # -> half spatial size

    # (3) Downsampling block 3
    c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    c3 = BatchNormalization()(c3)
    p3 = MaxPooling2D((2, 2))(c3)  # -> half spatial size

    # ----- BOTTLENECK -----
    c4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    c4 = BatchNormalization()(c4)
    # (No more downsampling here, it's the deepest part.)

    # ----- DECODER -----
    # (1) Upsampling from bottleneck to block 3
    u5 = UpSampling2D((2, 2))(c4)      # upsample: spatial size x2
    u5 = Conv2D(256, (2, 2), activation='relu', kernel_initializer='he_normal', padding='same')(u5)
    # Skip connection with c3
    u5 = concatenate([u5, c3])
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u5)
    c5 = BatchNormalization()(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    c5 = BatchNormalization()(c5)

    # (2) Upsampling from block 3 to block 2
    u6 = UpSampling2D((2, 2))(c5)
    u6 = Conv2D(128, (2, 2), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    # Skip connection with c2
    u6 = concatenate([u6, c2])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = BatchNormalization()(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    c6 = BatchNormalization()(c6)

    # (3) Upsampling from block 2 to block 1
    u7 = UpSampling2D((2, 2))(c6)
    u7 = Conv2D(64, (2, 2), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    # Skip connection with c1
    u7 = concatenate([u7, c1])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = BatchNormalization()(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    c7 = BatchNormalization()(c7)

    # ----- OUTPUT LAYER -----
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c7)

    # Build the final Model
    model = Model(inputs=inputs, outputs=outputs, name='unet_256filters')
    return model



def unet_model(height, width, channels):
    inputs = Input((height, width, channels))

    # Encoder
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    c3 = BatchNormalization()(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c4 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    c4 = BatchNormalization()(c4)

    # Decoder
    u5 = UpSampling2D((2, 2))(c4)
    u5 = Conv2D(64, (2, 2), activation='relu', kernel_initializer='he_normal', padding='same')(u5)
    u5 = concatenate([u5, c3])
    c5 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u5)
    c5 = BatchNormalization()(c5)
    c5 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    c5 = BatchNormalization()(c5)

    u6 = UpSampling2D((2, 2))(c5)
    u6 = Conv2D(32, (2, 2), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    u6 = concatenate([u6, c2])
    c6 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = BatchNormalization()(c6)
    c6 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    c6 = BatchNormalization()(c6)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = Conv2D(16, (2, 2), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    u7 = concatenate([u7, c1])
    c7 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = BatchNormalization()(c7)
    c7 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    c7 = BatchNormalization()(c7)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c7)

    # model = Model(inputs=[inputs], outputs=[outputs])
    # return model
    return Model(inputs=inputs, outputs=outputs, name='unet_simpel')






model = unet_model_256(height=400, width=400, channels=5)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()



# Create callbacks
checkpoint = ModelCheckpoint('shrub_UNET256fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT.tf',
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
    epochs=300,
    callbacks=[early_stopping, checkpoint]
)
