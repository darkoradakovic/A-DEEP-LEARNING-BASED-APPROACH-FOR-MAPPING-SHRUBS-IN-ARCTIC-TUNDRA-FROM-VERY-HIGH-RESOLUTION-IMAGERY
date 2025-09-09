
## ResNet50 model
# Darko Radakovic
# Montclair State University
# 20-03-25


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


# ADAPT paths
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
        # Full path PSHP  P1BS images
        pshp_image_path = os.path.join(pshp_dir, file_name)
        p1bs_image_path = pshp_image_path.replace('x_train', 'P1BS_files').replace('PSHP', 'P1BS')

        # Find y_train image
        image_path_y_train = None
        for yfilename in os.listdir(y_train_dir):
            if file_name[:-4] in yfilename and yfilename.endswith('.tif'):
                image_path_y_train = os.path.join(y_train_dir, yfilename)
                break  # Stop if found

        # If exists, proceed
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

        # each slice processing
        for s_pshp, s_p1bs, s_label in zip(slices_pshp, slices_p1bs, slices_label):
            ## Move axis to end for later processing
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






# ### METHOD 1. ResNet50 adjusted for 5 channels [works, unsharp but stable for Mean_p70]
# # Restnet50 is not producing sharp segmentation like the cnn_model3,
# # this is common experience when using large, generic backbones like ResNet50 for high-resolution semantic segmentation:
# # while the model can be more stable and robust, may appear less “sharp” (i.e.,  resulting segmentation maps have
# # coarser boundary detail) than a simpler, custom-designed CNN.

# from tensorflow.keras.models import load_model
# # base_model = load_model('/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/pycharm/above_nasa/resnet50_imagenet_savedmodel')
# base_model = load_model('/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/models/resnet50_imagenet_savedmodel')


# def select_first_3_channels(x):
#     # x shape: (None, 400, 400, 5)
#     # slice first 3 channels: x[:,:,:,0:3]
#     return x[:, :, :, :3]
#
# # Define the input for 5 channels
# input_5ch = Input(shape=(400,400,5))
#
# # Lambda layer to select first 3 channels
# three_channel_tensor = tf.keras.layers.Lambda(lambda x: x[:,:,:,0:3])(input_5ch)
#
#
# # Pass the 3-channel tensor into ResNet50
# x = base_model(three_channel_tensor)
#
# x = tf.keras.layers.UpSampling2D((2,2))(x)   # 13x13 -> 26x26
# x = tf.keras.layers.Conv2D(512, (3,3), padding='same', activation='relu')(x)
# x = tf.keras.layers.UpSampling2D((2,2))(x)   # 26x26 -> 52x52
# x = tf.keras.layers.Conv2D(256, (3,3), padding='same', activation='relu')(x)
# x = tf.keras.layers.UpSampling2D((2,2))(x)   # 52x52 -> 104x104
# x = tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
# x = tf.keras.layers.UpSampling2D((2,2))(x)   # 104x104 -> 208x208
# x = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
# x = tf.keras.layers.UpSampling2D((2,2))(x)   # 208x208 -> 416x416
# x = tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu')(x)
# # Now we have ~416x416. We can crop to 400x400 if needed:
# x = tf.keras.layers.Cropping2D(((8,8),(8,8)))(x) # Crop to 400x400
# x = tf.keras.layers.Conv2D(1, (1,1), activation='sigmoid')(x)
#
# # Create the final model
# model_5ch = Model(inputs=input_5ch, outputs=x)
#
# # Compile the model (for demonstration)
# model_5ch.compile(optimizer='adam', loss='binary_crossentropy')






### METHOD 2.  ResNet50 + U-Net-Style Decode

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (
    Input, Conv2D, Conv2DTranspose, UpSampling2D, Activation,
    BatchNormalization, concatenate, Lambda
)
from tensorflow.keras.models import Model

from tensorflow.keras.models import load_model

# RESNET PRETRAINED MODEL
# resnet_model = tf.keras.models.load_model('/Volumes/OWC Express 1M2/nasa_above/models/resnet50_imagenet_savedmodel_pretrained')
resnet_model = tf.keras.models.load_model('/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/models/resnet50_imagenet_savedmodel')

# resnet_model.summary()


def build_resnet50_unet(resnet_model, input_shape=(400, 400, 5)):
    """
    ImageNet-Pretrained ResNet50 encoder + U-Net-like decoder
    1. 5-channel input, extract 3 channels.
    2.
    3. Extracts skip connection outputs from specific ResNet layers.
    4. Build decoder with Conv2DTranspose / UpSampling2D, merging skip connections.

    Output:
      model_5ch: semantic segmentation.
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
    # Let ResNet50 manage the input_3ch via input_tensor:
    # Only first 3 channels.
    # Set include_top=False to get feature map output.
    # base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=x_3ch)
    # resnet_model = ResNet50(weights='imagenet', include_top=False, input_tensor=x_3ch)
    # encode_output = resnet_model(x_3ch)

    # Skip connections:
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

    #### 3. U-Net DECODER ####
    # Simple decoder -> Conv2DTranspose (or UpSampling2D)
    # gradually upsample. We'll merge skip connections by "concatenate".

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


# if __name__ == "__main__":
#     # Example usage
#     model_5ch = build_resnet50_unet(resnet_model=resnet_model, input_shape=(400, 400, 5))
#     model_5ch.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     model_5ch.summary()
#
#     # model_5ch.fit(train_dataset, epochs=50, validation_data=val_dataset)

# resnet_model = tf.keras.models.load_model("resnet50_imagenet_savedmodel")
model_5ch = build_resnet50_unet(resnet_model=resnet_model, input_shape=(400, 400, 5))
model_5ch.compile(optimizer='adam', loss='binary_crossentropy')
model_5ch.summary()








# checkpoint = ModelCheckpoint('shrub_RESNET50_pretrained_imagenet_weights_trainingdata3_RADnor_400px_PSHP_P1BS_v2_ADAPT.tf', monitor='val_loss', verbose=1, save_best_only=True)
# checkpoint = ModelCheckpoint('shrub_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_P1BS_v3_ADAPT.tf', monitor='val_loss', verbose=1, save_best_only=True)
checkpoint = ModelCheckpoint('shrub_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_b0_b3_v3_ADAPT.tf', monitor='val_loss', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)


# Train the model
history = model_5ch.fit(
    train_dataset,
    epochs=100,
    validation_data=val_dataset,
    callbacks=[checkpoint, early_stopping]
)











#### TEST MODEL

import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from functionsdarko import (slice_image,combine_tiles_to_large_image_predictionsoneyear,
                            rasterize_shapefile_to_binary
                            )

from skimage.filters import threshold_yen
# from sklearn.metrics import confusion_matrix
from skimage.filters import threshold_otsu
from skimage.filters import threshold_li

def load_image(image_path):
    with rasterio.open(image_path) as src:
        image = src.read()
        # return np.moveaxis(image, 0, -1), src.meta  # Reorder to (height, width, bands)
        return image


# shrub_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_v6_ADAPT_ep14.tf
model = load_model(
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/models/shrub_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_v6_ADAPT_ep14.tf'
)

# v1 Resnet model ADAPT ep3
model = load_model(
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/models/shrub_RESNET50_512fil_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_ep3.tf'
    # custom_objects={'f1_score': f1_score}
)


# v2 Resnet50 pretrained model ADAPT ep21 [BAD but oke metrics]
model = load_model(
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/models/shrub_RESNET50_pretrained_imagenet_weights_trainingdata3_RADnor_400px_PSHP_P1BS_v2_ADAPT_ep21.tf'
    # custom_objects={'f1_score': f1_score}
)

# v3 Resnet50 pretrained model ADAPT ep21 [BAD but oke metrics] shrub_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_P1BS_v3_ADAPT_ep8
model = load_model(
    '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/models/shrub_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_P1BS_v3_ADAPT_ep8.tf'
    # custom_objects={'f1_score': f1_score}
)



# ms1a1
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/QB02_20050724225626_1010010004654800_PSHP_P001_NT_2000m_area1.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20130820230414_1030010025076500_PSHP_P003_NT_2000m_area1.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20140707224109_1030010033485600_PSHP_P007_NT_2000m_area1.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV03_20170813230621_1040010031CAA600_PSHP_P003_NT_2000m_area1.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20200716223511_10300100A97DD900_PSHP_P003_NT_2000m_area1.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20220619223632_10300100D5894700_PSHP_P007_NT_2000m_area1.tif'

# ms6a2
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/QB02_20090718220421_1010010009F45F00_PSHP_P001_NT_2000m_area2.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20110625223400_103001000B99EE00_PSHP_P006_NT_2000m_area2.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/QB02_20130703205900_1010010011A29000_PSHP_P003_NT_2000m_area2.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV03_20160714215120_104001001F9EF500_PSHP_P001_NT_2000m_area2.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV03_20170811223418_1040010031305A00_PSHP_P001_NT_2000m_area2.tif'

# ms10a1
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/QB02_20050724225630_1010010004654800_PSHP_P002_NT_2000m_area1.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20140707224108_1030010033485600_PSHP_P006_NT_2000m_area1.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV03_20170813230622_1040010031CAA600_PSHP_P004_NT_2000m_area1.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train/WV02_20220619223631_10300100D5894700_PSHP_P006_NT_2000m_area1.tif'


# test site out of training data ms4a12
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/multisites/TIMESERIES/timeseries_ms4/timeseries_multisite4_areas/timeseries_multisite4_1_240323_202958/QB02_20020729211432_1010010000E40D00_P001_area12_-1738391_3976125/QB02_20020729211432_1010010000E40D00_PSHP_P001_NT_2000m_area12.tif'
pshp_image_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/multisites/TIMESERIES/timeseries_ms4/timeseries_multisite4_areas/timeseries_multisite4_13_240323_204307/WV02_20180826223219_103001008265EA00_P005_area12_-1738391_3976125/WV02_20180826223219_103001008265EA00_PSHP_P005_NT_2000m_area12.tif'






pshp_image = load_image(pshp_image_path)
p1bs_image_path = pshp_image_path.replace('x_train','P1BS_files').replace('PSHP','P1BS')
# ndvi_image_path = pshp_image_path.replace('x_train','ndvi').replace('PSHP','NDVI')
# ndvi_image = load_image(ndvi_image_path)
#
# # water_mask = ndvi_image < 0.1
# water_mask = ndvi_image < 0  # captures more shrubs

## STEP 0
target_slice_size_meters = 200
slices_pshp, metadata = slice_image(pshp_image_path, target_slice_size_meters)
slices_p1bs, _ = slice_image(p1bs_image_path, target_slice_size_meters)
# stacking PSHP + P1BS
slices_pshp = np.stack(slices_pshp)
slices_p1bs = np.stack(slices_p1bs)
slices_pshp = np.moveaxis(slices_pshp, 1, -1)
slices_p1bs = np.moveaxis(slices_p1bs, 1, -1)
x_train = np.concatenate((slices_pshp, slices_p1bs), axis=-1)  # x_train now has shape (num_samples, height, width, 5)
# y_train = y_train.astype('float32')
# Normalize
x_train = x_train / 255.0



# ## predictions on stacked PSHP P1BS
# predictions = model.predict(x_train,  verbose=1)
# combined_image, transformation_info = combine_tiles_to_large_image_predictionsoneyear(predictions, metadata)
#
#
# plt.figure(figsize=(8, 8))
# plt.imshow(np.squeeze(combined_image))
# plt.show(block=True)


# predictions 512fil model (wettundra cnn_model3)
predictions = []
for batch in np.array_split(x_train, 10):  # Split into smaller chunks
    predictions_batch = model.predict(batch, verbose=1)
    predictions.append(predictions_batch)
predictions = np.concatenate(predictions, axis=0)

combined_image, transformation_info = combine_tiles_to_large_image_predictionsoneyear(predictions, metadata)


plt.figure(figsize=(8, 8))
plt.imshow(np.squeeze(combined_image))
plt.show(block=True)
#
# combined_predictionimage_p70 = combined_image > np.percentile(combined_image,70)
# combined_predictionimage_p80 = combined_image > np.percentile(combined_image,80)
# combined_predictionimage_p90 = combined_image > np.percentile(combined_image,90)
# combined_predictionimage_p95 = combined_image > np.percentile(combined_image,95)
# combined_predictionimage_p99 = combined_image > np.percentile(combined_image,99)
# combined_predictionimage_otsu = combined_image > threshold_otsu(combined_image)
# combined_predictionimage_yen = combined_image > threshold_yen(combined_image)


# Calculate thresholds for each image
thresholds_accumulated = []
method = 'MEAN_p80'
for idx, pred in enumerate(predictions):
    pred_flat = pred.flatten()
    pred_flat = pred_flat[~np.isnan(pred_flat)]  # Remove NaN values
    if method == 'MEAN_otsu':   # method='MEAN_otsu'
        thresholds_accumulated.append(threshold_otsu(pred_flat))
    elif method == 'MEAN_yen':
        thresholds_accumulated.append(threshold_yen(pred_flat))
    elif method == 'MEAN_li':
        thresholds_accumulated.append(threshold_li(pred_flat))
    elif method == 'MEAN_p70':
        thresholds_accumulated.append(np.percentile(pred_flat, 70))
    elif method == 'MEAN_p80':
        thresholds_accumulated.append(np.percentile(pred_flat, 80))
    elif method == 'MEAN_90':
        thresholds_accumulated.append(np.percentile(pred_flat, 90))
    else:
        thresholds_accumulated.append(np.nan)
threshold_mean = np.mean(thresholds_accumulated)
combined_predictionimage_MEAN_p70 = combined_image > threshold_mean


# plt.figure(figsize=(8, 8))
# plt.imshow(np.squeeze(combined_predictionimage_MEAN_p70))
# plt.show(block=True)
#
# plt.figure(figsize=(8, 8))
# plt.imshow(np.squeeze(combined_predictionimage_p95))
#
# plt.figure(figsize=(8, 8))
# plt.imshow(np.squeeze(combined_predictionimage_p99))
#
#
# plt.figure(figsize=(8, 8))
# plt.imshow(np.squeeze(combined_predictionimage_yen))
#
#
# plt.figure(figsize=(8, 8))
# plt.imshow(np.squeeze(combined_predictionimage_otsu))
#
#
#
#
#
# # v2 shrub_RESNET50_pretrained_imagenet_weights_trainingdata3_RADnor_400px_PSHP_P1BS_v2_ADAPT_ep21
# file_namep70 = pshp_image_path.split('/')[-1].replace('.tif','_shrub_RESNET50_pretrained_imagenet_weights_trainingdata3_RADnor_400px_PSHP_P1BS_v2_ADAPT_ep21_p70.tif')
# file_namep80 = pshp_image_path.split('/')[-1].replace('.tif','_shrub_RESNET50_pretrained_imagenet_weights_trainingdata3_RADnor_400px_PSHP_P1BS_v2_ADAPT_ep21_p80.tif')
# file_namep90 = pshp_image_path.split('/')[-1].replace('.tif','_shrub_RESNET50_pretrained_imagenet_weights_trainingdata3_RADnor_400px_PSHP_P1BS_v2_ADAPT_ep21_p90.tif')
# file_namep95 = pshp_image_path.split('/')[-1].replace('.tif','_shrub_RESNET50_pretrained_imagenet_weights_trainingdata3_RADnor_400px_PSHP_P1BS_v2_ADAPT_ep21_p95.tif')

# shrub_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_v6_ADAPT_ep14.tf
file_namep70 = pshp_image_path.split('/')[-1].replace('.tif','_shrub_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_v6_ADAPT_ep14_p70.tif')

# # v3 shrub_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_P1BS_v3_ADAPT_ep8
# file_namep70 = pshp_image_path.split('/')[-1].replace('.tif','_shrub_RESNET50_pretrained_imagenet_weights_UNETdecoder_trainingdata3_RADnor_400px_PSHP_P1BS_v3_ADAPT_ep8_p70.tif')


output_folder = "/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/y_train2_shrubs_p70_v2"


output_image_path = os.path.join(output_folder, file_namep70)
print(output_image_path)

# Save using rasterio
with rasterio.open(pshp_image_path) as src:
    profile = src.profile
    profile.update(dtype=rasterio.uint8, count=1)
    with rasterio.open(output_image_path, 'w', **profile) as dst:
        ## -------->>>>> MANUALLY CHANGE ------------------
        # dst.write(modified_binary_image3.reshape(1,4000,4000))   ## -------->>>>> MANUALLY CHANGE
        # dst.write(modified_binary_image2.reshape(1, 4000, 4000))  ## -------->>>>> MANUALLY CHANGE
        dst.write(combined_predictionimage_MEAN_p70.reshape(1, 4000, 4000))  ## -------->>>>> MANUALLY CHANGE SMOOTH






