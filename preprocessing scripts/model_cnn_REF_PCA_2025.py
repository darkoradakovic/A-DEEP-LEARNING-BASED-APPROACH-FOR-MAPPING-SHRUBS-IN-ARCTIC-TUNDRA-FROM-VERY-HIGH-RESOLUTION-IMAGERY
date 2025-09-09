



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
# ref_dir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train_REF'
# pca_dir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train_PCA'
# ndviref_dir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train_NDVI_REF'
# y_train_dir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/y_train2_shrubs_p70_v2/final'

# shrubs
pshp_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/x_train'
p1sb_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/P1BS_files'
ref_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/x_train_REF'
pca_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/x_train_PCA'
ndviref_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/x_train_NDVI_REF'
y_train_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/y_train2_shrubs_p70_v2/final'




### MEMORY REDUCUCTION LOAD IMAGES

# Collect image triplets
image_stackpaths = []

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

        # REF
        image_path_ref = None
        for yfilename in os.listdir(ref_dir):
            if file_name[:-4] in yfilename and yfilename.endswith('.tif'):
                image_path_ref = os.path.join(ref_dir, yfilename)
                break
        # PCA
        image_path_pca = None
        for yfilename in os.listdir(pca_dir):
            if file_name[:-4] in yfilename and yfilename.endswith('.tif'):
                image_path_pca = os.path.join(pca_dir, yfilename)
                break
        # NDVI-REF
        image_path_ndviref = None
        for yfilename in os.listdir(ndviref_dir):
            if file_name[:-4] in yfilename and yfilename.endswith('.tif'):
                image_path_ndviref = os.path.join(ndviref_dir, yfilename)
                break

        # If corresponding y_train image exists, proceed
        if image_path_y_train:
            image_stackpaths.append((pshp_image_path, p1bs_image_path, image_path_ref, image_path_pca, image_path_ndviref, image_path_y_train))


from sklearn.model_selection import train_test_split

train_stackpaths, val_stackpaths = train_test_split(image_stackpaths, test_size=0.1, random_state=42)




# ---->
# ---- ---->
# ---- ---- ---->
# ---- ---- ---- ---->
# ---- ---- ---- ---- ---->
# ---- ---- ---- ---- ---- CHANGE HERE>
# ---- ---- ---- ---- ---->
# ---- ---- ---- ---->
# ---- ---- ---->
# ---- ---->
# ---->

# Define the data generator function
def data_generator_from_stackpaths(triplets, target_slice_size_meters):
    for pshp_image_path, p1bs_image_path, ref_image_path, pca_image_path, ndviref_image_path, image_path_y_train in triplets:
        # Slice PSHP and P1BS images
        slices_pshp, _ = slice_image(pshp_image_path, target_slice_size_meters)
        slices_p1bs, _ = slice_image(p1bs_image_path, target_slice_size_meters)
        slices_ref, _ = slice_image(ref_image_path, target_slice_size_meters)
        slices_pca, _ = slice_image(pca_image_path, target_slice_size_meters)
        slices_ndviref, _ = slice_image(ndviref_image_path, target_slice_size_meters)
        slices_label, _ = slice_image(image_path_y_train, target_slice_size_meters)

        # For each slice, process and yield
        for s_pshp, s_p1bs, s_ref, s_pca, s_ndviref, s_label in zip(slices_pshp, slices_p1bs, slices_ref, slices_pca, slices_ndviref, slices_label):
            # Move axis to have channels at the end
            s_pshp = np.moveaxis(s_pshp, 0, -1)  # Shape: (height, width, channels)
            s_p1bs = np.moveaxis(s_p1bs, 0, -1)  # Shape: (height, width, channels)
            s_ref = np.moveaxis(s_ref, 0, -1)  # Shape: (height, width, channels)
            s_pca = np.moveaxis(s_pca, 0, -1)  # Shape: (height, width, channels)
            s_ndviref = np.moveaxis(s_ndviref, 0, -1)  # Shape: (height, width, channels)

            # ---- ---- ---- ---- ---- CHANGE HERE>
            # ---- ---- ---- ---- ---- CHANGE HERE>
            # ---- ---- ---- ---- ---- CHANGE HERE>
            # ---- ---- ---- ---- ---- CHANGE HERE>

            # Concatenate PSHP and P1BS along the channel axis to form x_slice
            # x_slice = np.concatenate((s_pshp, s_p1bs, s_ref, s_pca, s_ndviref), axis=-1)  # Shape: (height, width, 5)
            # x_slice = np.concatenate((s_ref, s_p1bs), axis=-1)  # Shape: (height, width, 5)
            # x_slice = np.concatenate((s_ref, s_ndviref), axis=-1)  # Shape: (height, width, 5)
            x_slice = np.concatenate((s_ref, s_pca, s_ndviref), axis=-1)  # Shape: (height, width, 8)
            x_slice = x_slice / 255.0

            # Process y_slice
            y_slice = np.moveaxis(s_label, 0, -1).astype('float32')  # Shape: (height, width, 1)

            yield x_slice, y_slice




batch_size = 8  # cnn_model3 (test wet tundra)

# Create datasets with output_signature
train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator_from_stackpaths(train_stackpaths, target_slice_size_meters),
    output_signature=(
        # tf.TensorSpec(shape=(400, 400, 5), dtype=tf.float32),
        tf.TensorSpec(shape=(400, 400, 8), dtype=tf.float32),
        tf.TensorSpec(shape=(400, 400, 1), dtype=tf.float32)
    )
)

val_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator_from_stackpaths(val_stackpaths, target_slice_size_meters),
    output_signature=(
        # tf.TensorSpec(shape=(400, 400, 5), dtype=tf.float32),
        tf.TensorSpec(shape=(400, 400, 8), dtype=tf.float32),
        tf.TensorSpec(shape=(400, 400, 1), dtype=tf.float32)
    )
)

# Configure datasets
train_dataset = train_dataset.shuffle(buffer_size=1000)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

val_dataset = val_dataset.batch(batch_size)
val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)





##### ---- ---- ---- ----
##### ---- ---- ---- ----  ResNet50 + U-Net-Style Decode
##### ---- ---- ---- ----

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (
    Input, Conv2D, Conv2DTranspose, UpSampling2D, Activation,
    BatchNormalization, concatenate, Lambda
)
from tensorflow.keras.models import Model

from tensorflow.keras.models import load_model
# resnet_model = tf.keras.models.load_model('/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/pycharm/above_nasa/resnet50_imagenet_savedmodel')
resnet_model = tf.keras.models.load_model('/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/models/resnet50_imagenet_savedmodel')


def build_resnet50_unet(resnet_model, input_shape=(400, 400, 5)):
    """
    Build a U-Net-like model with ResNet50 (ImageNet-pretrained) as an encoder.
    1. Takes 5-channel input, slices or combines it into 3 channels.
    2. Feeds it into ResNet50.
    3. Extracts skip connection outputs from chosen ResNet layers.
    4. Builds a decoder with Conv2DTranspose or UpSampling2D, merging skip connections.

    Returns:
      model_5ch: a Keras Model for semantic segmentation.
    """

    #### 1. INPUT LAYER (5-ch) -> LAMBDA to (3-ch) ####
    # input_5ch = tf.keras.layers.Input(shape=input_shape, name='input_5ch')
    # # For example, slice first 3 channels:
    # # x_3ch = Lambda(lambda x: x[:, :, :, 0:3])(input_5ch)
    # x_3ch = tf.keras.layers.Lambda(lambda x: x[:,:,:,1:4])(input_5ch)

    new_input_5ch = Input(shape=input_shape, name='new_input_5ch')
    # three_ch = Lambda(lambda x: x[:, :, :, 1:4], name='slice_3ch')(new_input_5ch)
    three_ch = Lambda(lambda x: x[:, :, :, 0:3], name='slice_3ch')(new_input_5ch)

    #### 2. RESNET50 ENCODER (3-ch) ####
    # We'll let ResNet50 manage the input_3ch via input_tensor:
    # Note that this uses pre-trained ImageNet weights for the 3 channels.
    # We set include_top=False to get a feature map output.
    # base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=x_3ch)
    # resnet_model = ResNet50(weights='imagenet', include_top=False, input_tensor=x_3ch)
    # encode_output = resnet_model(x_3ch)

    # We'll collect some of these to use as skip connections:
    conv1_relu = resnet_model.get_layer('conv1_relu').output
    conv2_out = resnet_model.get_layer('conv2_block3_out').output
    conv3_out = resnet_model.get_layer('conv3_block4_out').output
    conv4_out = resnet_model.get_layer('conv4_block6_out').output
    conv5_out = resnet_model.get_layer('conv5_block3_out').output

    encoder_out = resnet_model.get_layer('conv5_block3_out').output  # (None,13,13,2048)
    resnet_encoder = Model(
        inputs=resnet_model.input,
        outputs=[conv1_relu, conv2_out, conv3_out, conv4_out, conv5_out],
        name='resnet50_encoder'
    )

    skip1, skip2, skip3, skip4, enc_out = resnet_encoder(three_ch)

    #### 3. DECODER ####
    # We'll build a simple decoder that uses Conv2DTranspose (or UpSampling2D)
    # to gradually upsample. We'll merge skip connections by "concatenate".

    x = UpSampling2D((2, 2), interpolation='bilinear')(enc_out)  # 13->26
    x = Conv2D(1024, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    # Crop 26->25 if needed
    x = Cropping2D(((0, 1), (0, 1)))(x)
    x = concatenate([x, skip4])

    x = UpSampling2D((2, 2))(x)  # 25->50
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = concatenate([x, skip3])

    x = UpSampling2D((2, 2))(x)  # 50->100
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = concatenate([x, skip2])

    x = UpSampling2D((2, 2))(x)  # 100->200
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = concatenate([x, skip1])

    x = UpSampling2D((2, 2))(x)  # 200->400
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    # Output
    final = Conv2D(1, (1, 1), padding='same', activation='sigmoid', name='final_mask')(x)

    # 5) Build a top-level model from new_input_5ch to final
    unet_model = Model(inputs=new_input_5ch, outputs=final, name='ResNet50_U-Net_5ch')
    return unet_model

# resnet_model = tf.keras.models.load_model("resnet50_imagenet_savedmodel")
model_5ch = build_resnet50_unet(resnet_model=resnet_model, input_shape=(400, 400, 5))
model_5ch.compile(optimizer='adam', loss='binary_crossentropy')
model_5ch.summary()

# checkpoint = ModelCheckpoint('shrub_RESNET50_pretrained_imagenet_weights_trainingdata3_RADnor_400px_PSHP_P1BS_v2_ADAPT.tf', monitor='val_loss', verbose=1, save_best_only=True)
# checkpoint = ModelCheckpoint('shrub_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_P1BS_v3_ADAPT.tf', monitor='val_loss', verbose=1, save_best_only=True)
checkpoint = ModelCheckpoint('shrub_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_REF_PSHP_400px_b0_b3_v1_ADAPT.tf', monitor='val_loss', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)


# Train the model
history = model_5ch.fit(
    train_dataset,
    epochs=100,
    validation_data=val_dataset,
    callbacks=[checkpoint, early_stopping]
)







##### ---- ---- ---- ----
##### ---- ---- ---- ---- CNN 512fil
##### ---- ---- ---- ----

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization, Activation, Cropping2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

def cnn_model3(height, width, channels, dropout_rate=0.3, l2_lambda=0.001):
# def cnn_model3(height, width, channels, dropout_rate=0.5, l2_lambda=0.01):
    x_input = Input(shape=(height, width, channels))

    # Convolutional Layer 1
    x = Conv2D(32, (3, 3), padding='same', activation='relu',
               kernel_regularizer=l2(l2_lambda))(x_input)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # Convolutional Layer 2
    x = Conv2D(64, (3, 3), padding='same', activation='relu',
               kernel_regularizer=l2(l2_lambda))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # Convolutional Layer 3
    x = Conv2D(128, (3, 3), padding='same', activation='relu',
               kernel_regularizer=l2(l2_lambda))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # Convolutional Layer 4
    x = Conv2D(256, (3, 3), padding='same', activation='relu',
               kernel_regularizer=l2(l2_lambda))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # **New Convolutional Layer 5 with 512 filters**
    x = Conv2D(512, (3, 3), padding='same', activation='relu',
               kernel_regularizer=l2(l2_lambda))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # Output Layer
    outputs = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x)

    return Model(inputs=x_input, outputs=outputs, name='cnn_model3_512')




# model = cnn_model3(height=400, width=400, channels=5)
model = cnn_model3(height=400, width=400, channels=8)   #PCA+NDVI


opt = tf.keras.optimizers.legacy.Adam(learning_rate=1e-4)

# model.compile(optimizer=Adam(learning_rate=1e-4),
model.compile(optimizer=opt,
              loss='binary_crossentropy',
              # loss=loss_fn,
              metrics=['accuracy'])


# checkpoint = ModelCheckpoint('shrub_cnn_model2_256fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS.tf', monitor='val_loss', verbose=1, save_best_only=True)
# checkpoint = ModelCheckpoint('shrub_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS.tf', monitor='val_loss', verbose=1, save_best_only=True)
# checkpoint = ModelCheckpoint('shrub_CNN_512fil_5layers_trainingdata3_REF_400px_PSHP_P1BS_5bands.tf', monitor='val_loss', verbose=1, save_best_only=True)
# checkpoint = ModelCheckpoint('shrub_CNN_512fil_5layers_trainingdata3_REF_400px_PSHP_NDVI_v2_5bands.tf', monitor='val_loss', verbose=1, save_best_only=True)
checkpoint = ModelCheckpoint('shrub_CNN_512fil_5layers_trainingdata3_REF_400px_PSHP_PCA_NDVI_v3_8bands.tf', monitor='val_loss', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)



# Train the model
history = model.fit(
    train_dataset,
    epochs=100,
    validation_data=val_dataset,
    callbacks=[checkpoint, early_stopping]
)





##### ---- ---- ---- ----
##### ---- ---- ---- ---- VGG19
##### ---- ---- ---- ----

vgg19_path = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/models/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'


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
    Creates a U-Net with a VGG19 encoder. The top-level input has shape (400,400,5),
    but only 3 out of those 5 channels get passed to VGG19. The rest are effectively
    ignored by the encoder.

    Steps:
      1) Input of shape (400,400,5).
      2) Slice out the first 3 channels -> shape (400,400,3).
      3) Pass that to a standard VGG19 (include_top=False).
      4) Grab skip outputs from the blocks.
      5) Build a U-Net style decoder to upsample + skip-conn.
      6) Output is a 1-channel segmentation mask.
    """

    ### (A) Top-level input: (400,400,5)
    inputs_5ch = Input(shape=input_shape, name="input_5ch")

    ### (B) Slice out 3 channels
    # For example, channels [0..2] in the last dimension:
    three_ch = Lambda(
        lambda x: x[:,:,:,0:3],
        name="slice_3ch"
    )(inputs_5ch)

    ### (C) Create a plain VGG19 (no top), with shape = (None, None, 3).
    # We do *not* pass input_tensor here. We'll handle skip outputs manually.
    vgg_base = VGG19(
        weights=None,
        include_top=False,
        input_shape=(None, None, 3)  # dynamically sized
    )
    # Load the pretrained "notop" weights from a local .h5
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

    ### Build an encoder sub-model to get skip outputs:
    encoder_sub = Model(
        inputs=vgg_base.input,   # shape(None,None,3)
        outputs=[s1, s2, s3, s4, encoder_out],
        name='vgg19_encoder_sub'
    )

    ### Actually run the 3-ch tensor through that sub-model:
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

    # final mask => 1 channel, sigmoid
    outputs = Conv2D(1, (1,1), padding='same', activation='sigmoid')(x)

    ### Build the top-level:
    #   input = (400,400,5), output = (400,400,1)
    model_5ch = Model(inputs=inputs_5ch, outputs=outputs, name="vgg19_unet_5ch")
    return model_5ch



import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Build the model
model = build_vgg_unet_5channels(
        vgg_weights_path=vgg19_path,
        input_shape=(400,400,5)
    )

model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])

# Create callbacks
checkpoint = ModelCheckpoint('shrub_VGG19_pretrained_imagenet_weights_UNETdecoder_trainingdata3_REF_400px_PSHP_b0_b3_v1_ADAPT.tf',
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






##### ---- ---- ---- ----
##### ---- ---- ---- ---- UNET 256fil
##### ---- ---- ---- ----



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



# model = unet_model_256(height=400, width=400, channels=5)
model = unet_model_256(height=400, width=400, channels=8)   # PCA + NDVI
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()



# Create callbacks
# checkpoint = ModelCheckpoint('shrub_UNET256fil_trainingdata3_REF_400px_PSHP_P1BS_v1_ADAPT.tf',
# checkpoint = ModelCheckpoint('shrub_UNET256fil_trainingdata3_REF_400px_PSHP_NDVI_v2_ADAPT.tf',
checkpoint = ModelCheckpoint('shrub_UNET256fil_trainingdata3_REF_400px_PSHP_PCA_NDVI_v3_ADAPT.tf',
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



