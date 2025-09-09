
## CNN 512-fil model
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
from tensorflow.keras.models import load_model
from skimage.transform import resize


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


# from functionsdarko import (slice_image,combine_tiles_to_large_image_predictionsoneyear
#                             )


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

# Parameters to save the slices
# save_directory = "/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/simple_shrub_CNN/ms10a1_QB02_20050724225630_1010010004654800_PSHP_P002_NT_2000m_area1/x_train_100m"  # Directory where slices will be saved

# # shrubs  training data local paths
# pshp_dir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train'
# p1sb_dir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/P1BS_files'
# y_train_dir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/y_train2_shrubs_p70_v2/final'

# ADAPT paths
pshp_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/x_train'
p1sb_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/P1BS_files'
y_train_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/final'

# # wettundra
# y_train_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/y_train_wettundra3'

# # lakes and rivers
# y_train_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/y_train_lakes_and_rivers'


# x_train_slices = []
# y_train_slices = []
#
#
# ### TOO HEAVY ON MEMORY
# count = 0
# for file_name in os.listdir(pshp_dir):
#     if file_name.endswith('.tif'):
#         # Full path to the PSHP and P1BS images
#         pshp_image_path = os.path.join(pshp_dir, file_name)
#         p1bs_image_path = pshp_image_path.replace('x_train', 'P1BS_files').replace('PSHP', 'P1BS')
#
#         # Find corresponding y_train image
#         image_path_y_train = None
#         for yfilename in os.listdir(y_train_dir):
#             if file_name[:-4] in yfilename and yfilename.endswith('.tif'):
#                 image_path_y_train = os.path.join(y_train_dir, yfilename)
#                 break  # Stop searching once the correct label is found
#
#         # If corresponding y_train image exists, proceed
#         if image_path_y_train:
#             # Slice PSHP and P1BS images
#             slices_pshp, _ = slice_image(pshp_image_path, target_slice_size_meters)
#             slices_p1bs, _ = slice_image(p1bs_image_path, target_slice_size_meters)
#
#             # Stack PSHP and P1BS slices together along the channel dimension
#             slices_pshp = np.stack(slices_pshp)  # Shape: (num_slices, channels, height, width)
#             slices_p1bs = np.stack(slices_p1bs)
#
#             # Move axis to have channels at the end
#             slices_pshp = np.moveaxis(slices_pshp, 1, -1)  # Shape: (num_slices, height, width, channels)
#             slices_p1bs = np.moveaxis(slices_p1bs, 1, -1)  # Shape: (num_slices, height, width, channels)
#
#             # Concatenate PSHP and P1BS along the channel axis to form x_train
#             x_train = np.concatenate((slices_pshp, slices_p1bs), axis=-1)  # Shape: (num_slices, height, width, 5)
#             x_train = x_train / 255.0
#
#             # Slice the label image (y_train)
#             slices_label, _ = slice_image(image_path_y_train, target_slice_size_meters)
#             y_train = np.stack(slices_label)  # Shape: (num_slices, channels, height, width)
#             y_train = np.moveaxis(y_train, 1, -1)  # Shape: (num_slices, height, width, channels)
#             y_train = y_train.astype('float32')
#
#             # Append the slices to the list
#             x_train_slices.append(x_train)
#             y_train_slices.append(y_train)
#
#             count += 1
#             if count == 2:
#                 break
#
# x_train_all = np.concatenate(x_train_slices, axis=0)  # Shape: (total_slices, height, width, 5)
# y_train_all = np.concatenate(y_train_slices, axis=0)  # Shape: (total_slices, height, width, 1)


# # normal code
# x_temp, x_test, y_temp, y_test = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=42)
# x_train2, x_val, y_train2, y_val = train_test_split(x_temp, y_temp, test_size=0.125, random_state=42)  # 0.125 * 0.8 = 0.1



### MEMORY REDUCUCTION LOAD IMAGES
# Collect all image paths
image_triplets = []

for file_name in os.listdir(pshp_dir):
    if file_name.endswith('.tif'):
        # Full path PSHP + P1BS images
        pshp_image_path = os.path.join(pshp_dir, file_name)
        p1bs_image_path = pshp_image_path.replace('x_train', 'P1BS_files').replace('PSHP', 'P1BS')

        # Find y_train image
        image_path_y_train = None
        for yfilename in os.listdir(y_train_dir):
            if file_name[:-4] in yfilename and yfilename.endswith('.tif'):
                image_path_y_train = os.path.join(y_train_dir, yfilename)
                break

        # If corresponding y_train image exists, proceed
        if image_path_y_train:
            image_triplets.append((pshp_image_path, p1bs_image_path, image_path_y_train))


from sklearn.model_selection import train_test_split

train_triplets, val_triplets = train_test_split(image_triplets, test_size=0.1, random_state=42)

def generator(triplets, target_slice_size_meters):
    for pshp_image_path, p1bs_image_path, image_path_y_train in triplets:
        # Slice PSHP P1BS images
        slices_pshp, _ = slice_image(pshp_image_path, target_slice_size_meters)
        slices_p1bs, _ = slice_image(p1bs_image_path, target_slice_size_meters)

        # Slice (y_train)
        slices_label, _ = slice_image(image_path_y_train, target_slice_size_meters)

        # for each slice processing
        for s_pshp, s_p1bs, s_label in zip(slices_pshp, slices_p1bs, slices_label):
            # Move axis to end for later processing
            s_pshp = np.moveaxis(s_pshp, 0, -1)  # Shape: (height, width, channels)
            s_p1bs = np.moveaxis(s_p1bs, 0, -1)  # Shape: (height, width, channels)

            # Concatenate PSHP + P1BS
            x_slice = np.concatenate((s_pshp, s_p1bs), axis=-1)  # Shape: (height, width, 5)
            x_slice = x_slice / 255.0

            # Process y_slice
            y_slice = np.moveaxis(s_label, 0, -1).astype('float32')  # Shape: (height, width, 1)

            yield x_slice, y_slice




batch_size = 8  # cnn_model3 (test wet tundra)

# Training dataset
train_dataset = tf.data.Dataset.from_generator(
    lambda: generator(train_triplets, target_slice_size_meters),
    output_types=(tf.float32, tf.float32),
    output_shapes=((None, None, 5), (None, None, 1))
)

# Validation dataset
val_dataset = tf.data.Dataset.from_generator(
    lambda: generator(val_triplets, target_slice_size_meters),
    output_types=(tf.float32, tf.float32),
    output_shapes=((None, None, 5), (None, None, 1))
)


train_dataset = train_dataset.shuffle(buffer_size=1000)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

val_dataset = val_dataset.batch(batch_size)
val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)



## CLASS WEIGHTS
def compute_class_weights(dataset, num_batches=10):
    total_counts = np.array([0, 0], dtype=np.int64)
    for i, (x_batch, y_batch) in enumerate(dataset.take(num_batches)):
        y_batch = y_batch.numpy()
        counts = np.bincount(y_batch.astype(int).flatten(), minlength=2)
        total_counts += counts
    total_pixels = total_counts.sum()
    class_weights = total_pixels / (2 * total_counts)
    return class_weights

# weights
class_weights = compute_class_weights(train_dataset)
print("Computed class weights:", class_weights)

def weighted_binary_crossentropy(class_weights):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)  # Shape: [batch_size, height, width, 1]
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)  # Shape: [batch_size, height, width]
        y_true_squeezed = tf.squeeze(y_true, axis=-1)  # Shape: [batch_size, height, width]
        weight_vector = y_true_squeezed * class_weights[1] + (1. - y_true_squeezed) * class_weights[0]
        weighted_bce = weight_vector * bce
        return tf.reduce_mean(weighted_bce)
    return loss

loss_fn = weighted_binary_crossentropy(class_weights)


# ## Save figure
# class CustomModelCheckpointAndPlot(tf.keras.callbacks.Callback):
#     def __init__(self, filepath, save_freq_epochs):
#         super(CustomModelCheckpointAndPlot, self).__init__()
#         self.filepath = filepath
#         self.save_freq_epochs = save_freq_epochs
#         self.epoch_history = []
#         self.history = {}
#
#     def on_epoch_end(self, epoch, logs=None):
#         logs = logs or {}
#         self.epoch_history.append(epoch)
#         for k, v in logs.items():
#             self.history.setdefault(k, []).append(v)
#         if (epoch + 1) % self.save_freq_epochs == 0:
#             # Save the model weights
#             filepath = self.filepath.format(epoch=epoch + 1)
#             self.model.save_weights(filepath)
#             print(f'\nEpoch {epoch + 1}: saving model to {filepath}')
#             # Save the learning curves
#             self.plot_and_save(epoch + 1)
#
#     def plot_and_save(self, epoch):
#         # Plot training & validation accuracy values
#         plt.figure()
#         plt.plot(range(1, epoch + 1), self.history['accuracy'], label='Train Accuracy')
#         plt.plot(range(1, epoch + 1), self.history['val_accuracy'], label='Validation Accuracy')
#         plt.title(f'Model Accuracy up to Epoch {epoch}')
#         plt.ylabel('Accuracy')
#         plt.xlabel('Epoch')
#         plt.legend(loc='upper left')
#         # Save the figure
#         filename = self.filepath.format(epoch=epoch).replace('.h5', '_accuracy.png')
#         plt.savefig(filename)
#         plt.close()
#         # Plot training & validation loss values
#         plt.figure()
#         plt.plot(range(1, epoch + 1), self.history['loss'], label='Train Loss')
#         plt.plot(range(1, epoch + 1), self.history['val_loss'], label='Validation Loss')
#         plt.title(f'Model Loss up to Epoch {epoch}')
#         plt.ylabel('Loss')
#         plt.xlabel('Epoch')
#         plt.legend(loc='upper left')
#         # Save the figure
#         filename = self.filepath.format(epoch=epoch).replace('.h5', '_loss.png')
#         plt.savefig(filename)
#         plt.close()






### MODEL TRAIN ALL YEARS

#
# # old
# def f1_score(y_true, y_pred):
#     def recall(y_true, y_pred):
#         true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#         possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#         recall = true_positives / (possible_positives + K.epsilon())
#         return recall
#
#     def precision(y_true, y_pred):
#         true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#         predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#         precision = true_positives / (predicted_positives + K.epsilon())
#         return precision
#
#     precision = precision(y_true, y_pred)
#     recall = recall(y_true, y_pred)
#     return 2*((precision*recall)/(precision+recall+K.epsilon()))
#
#
#
# ## OLD CLASS IMBALANCE
# unique, counts = np.unique(y_train2, return_counts=True)
# class_distribution = dict(zip(unique, counts))
# print("Class distribution:", class_distribution)
#
# #
# weights = compute_class_weight('balanced', classes=np.unique(y_train2), y=y_train2.flatten())
# class_weights = {i: weights[i] for i in range(len(weights))}
#
# print("Class weights:", class_weights)
#


from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization, Activation, Cropping2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2



# VERY GOOD, USING FOR MULTIYEAR
def cnn_model2(height, width, channels, dropout_rate=0.3, l2_lambda=0.001):
# def cnn_model2(height, width, channels, dropout_rate=0.5, l2_lambda=0.01):
    x_input = Input(shape=(height, width, channels))

    # Conv2D Layer 1
    x = Conv2D(32, (3, 3), padding='same', activation='relu',
               kernel_regularizer=l2(l2_lambda))(x_input)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # Conv2D Layer 2
    x = Conv2D(64, (3, 3), padding='same', activation='relu',
               kernel_regularizer=l2(l2_lambda))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # Conv2D Layer 3
    x = Conv2D(128, (3, 3), padding='same', activation='relu',
               kernel_regularizer=l2(l2_lambda))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # Conv2D Layer 3
    x = Conv2D(256, (3, 3), padding='same', activation='relu',
               kernel_regularizer=l2(l2_lambda))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # Output
    outputs = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x)

    return Model(inputs=x_input, outputs=outputs, name='cnn_model2_256')


def cnn_model3(height, width, channels, dropout_rate=0.3, l2_lambda=0.001):
# def cnn_model3(height, width, channels, dropout_rate=0.5, l2_lambda=0.01):
    x_input = Input(shape=(height, width, channels))

    # Conv2D Layer1
    x = Conv2D(32, (3, 3), padding='same', activation='relu',
               kernel_regularizer=l2(l2_lambda))(x_input)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # Conv2D Layer  2
    x = Conv2D(64, (3, 3), padding='same', activation='relu',
               kernel_regularizer=l2(l2_lambda))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # Conv2D Layer 3
    x = Conv2D(128, (3, 3), padding='same', activation='relu',
               kernel_regularizer=l2(l2_lambda))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # Conv2D Layer 4
    x = Conv2D(256, (3, 3), padding='same', activation='relu',
               kernel_regularizer=l2(l2_lambda))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # Conv2D Layer  512 filters
    x = Conv2D(512, (3, 3), padding='same', activation='relu',
               kernel_regularizer=l2(l2_lambda))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # Output
    outputs = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x)

    return Model(inputs=x_input, outputs=outputs, name='cnn_model3_512')





# checkpoint_callback = CustomModelCheckpointAndPlot(
#     filepath='shrub_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS_v7_ep{epoch:02d}.tf',
#     save_freq_epochs=5  # Save every 5 epochs
# )

## simple CNN model MULTIYEAR 2005, 2014, 2017, 2022 ms10a1
# model = cnn_model2(height=400, width=400, channels=4)
# model = cnn_model2(height=400, width=400, channels=5)
model = cnn_model3(height=400, width=400, channels=5)


opt = tf.keras.optimizers.legacy.Adam(learning_rate=1e-4)

# model.compile(optimizer=Adam(learning_rate=1e-4),
model.compile(optimizer=opt,
              # loss='binary_crossentropy',
              loss=loss_fn,
              metrics=['accuracy'])

# checkpoint = ModelCheckpoint('shrub_cnn_model2_256fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS.tf', monitor='val_loss', verbose=1, save_best_only=True)
checkpoint = ModelCheckpoint('shrub_cnn_model3_512fil_5layers_trainingdata3_RADnor_400px_PSHP_P1BS.tf', monitor='val_loss', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

# batch_size = 16
# steps_per_epoch = len(x_train2) // batch_size
#
# history = model.fit(x_train2, y_train2,
#                     batch_size=batch_size,
#                     steps_per_epoch=steps_per_epoch,
#                     epochs=300,
#                     validation_data=(x_val, y_val),
#                     # class_weight=class_weight_dict,
#                     callbacks=[checkpoint, early_stopping])

# Train the model
history = model.fit(
    train_dataset,
    epochs=100,
    validation_data=val_dataset,
    callbacks=[checkpoint, early_stopping]
    # callbacks=[checkpoint_callback, early_stopping]
)


# # Evaluation
# loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
# # loss, accuracy, f1, iou = model.evaluate(x_test, y_test, verbose=1)
# print(f'Test loss: {loss}')
# print(f'Test accuracy: {accuracy}')
# # print(f'F1 Score: {f1}')
# # print(f'IoU Score: {iou}')







