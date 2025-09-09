
### ## --- --- VISIONTRANSFORMER MODELS --- --- 17-1-25

# replicating paper Tulan 2024, by using  DINOv2 Pretrained ViT Model and a Dense Prediction (DPT Decoder)

# pip install torch timm safetensors transformers

## 25-3-25
# USED this version to train WETTUNDRA AND LAKESRIVERS VIT MODELS



import os
import rasterio
from rasterio.windows import Window
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm                    # for the DINOv2-based backbone if available here
# or from your local library if you have a custom 'DINOv2' code
import safetensors.torch       # if your weights are in .safetensors
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import tensorflow as tf
# We'll read your (H,W,5)/(H,W,1) data from tf.data,
# then convert to np → torch.

from functionsdarko import (slice_image,combine_tiles_to_large_image_predictionsoneyear
                            )

S = '/Volumes/OWC Express 1M2/nasa_above/UNET_TRAINING_DATA_v2/y_train_wettundra3'

target_slice_size_meters = 200  ## in meter
#
# # shrubs
# pshp_dir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train'
# p1sb_dir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/P1BS_files'
# y_dir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/y_train2_shrubs_p70_v2/final'
pshp_dir = '/Volumes/OWC Express 1M2/nasa_above/UNET_TRAINING_DATA_v2/x_train'
p1sb_dir = '/Volumes/OWC Express 1M2/nasa_above/UNET_TRAINING_DATA_v2/P1BS_files'

# # # ADAPT
# pshp_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/x_train'
# p1sb_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/P1BS_files'
#y_train_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/final'
# y_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/y_train2_shrubs_p70_v2/final'

# # wettundra
# y_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/y_train_wettundra3'
y_dir = '/Volumes/OWC Express 1M2/nasa_above/UNET_TRAINING_DATA_v2/y_train_wettundra3'


# # # lakes and rivers
# y_dir = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/y_train_lakes_and_rivers'


# local_weights_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/models/VIT_pretrained_models--timm--vit_base_patch16_384.augreg_in21k_ft_in1k/snapshots/1c30cbc7ec35e68529522c0c2ac55692f83cff18/model.safetensors'
# local_weights_path = "/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/models/VIT_pretrained_models--timm--vit_base_patch16_384.augreg_in21k_ft_in1k/snapshots/1c30cbc7ec35e68529522c0c2ac55692f83cff18/model.safetensors"
## DINOv2
# local_weights_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/models/VIT_pretrained_models--timm--vit_large_patch14_dinov2.lvd142m/snapshots/2e99f078e7011d363e5e6647c7b0383e11788b68/model.safetensors'
local_weights_path = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/models/VIT_pretrained_models--timm--vit_large_patch14_dinov2.lvd142m/snapshots/2e99f078e7011d363e5e6647c7b0383e11788b68/model.safetensors'
# local_weights_path = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/models/shrub_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_48h.pth'


## Pretrained VIT
# import timm
# model = timm.create_model('vit_base_patch16_224', pretrained=True)
# model = timm.create_model('vit_base_patch16_384', pretrained=True)
# Load DINOv2 pretrained ViT-Large model
# model_vit = timm.create_model('vit_large_patch14_dinov2.lvd142m', pretrained=True)

# # 1) Create the model with pretrained=False so it doesn't try to download

# 2) Load your local weights:
# model = timm.create_model('vit_base_patch16_224', pretrained=False)
# state_dict = load_file(local_weights_path)
# model.load_state_dict(state_dict, strict=False)



##########################################################
# B) Slicing, reading (like you have)
##########################################################
def slice_image(image_path, target_slice_size_meters=200):
    """
    Returns slices: list of arrays shape [C, H, W], plus metadata if needed.
    """
    import rasterio
    from rasterio.windows import Window

    slices = []
    metadata = []
    with rasterio.open(image_path) as src:
        pixel_size = src.res[0]
        slice_size_pixels = int(target_slice_size_meters / pixel_size)
        for i in range(0, src.width, slice_size_pixels):
            for j in range(0, src.height, slice_size_pixels):
                w = min(slice_size_pixels, src.width - i)
                h = min(slice_size_pixels, src.height - j)
                window = Window(i, j, w, h)
                arr = src.read(window=window)  # shape (C, h, w)
                slices.append(arr)
                meta = src.meta.copy()
                meta.update({
                    'height': h,
                    'width': w,
                    'transform': rasterio.windows.transform(window, src.transform)
                })
                metadata.append(meta)
    return slices, metadata

def data_generator_from_triplets(image_triplets, target_slice_size_meters=200):
    """
    For each (pshp, p1bs, label), yields (x_slice, y_slice).
    x_slice => shape (H, W, 5)
    y_slice => shape (H, W, 1)
    """
    for pshp_path, p1bs_path, label_path in image_triplets:
        pshp_slices, _ = slice_image(pshp_path, target_slice_size_meters)
        p1bs_slices, _ = slice_image(p1bs_path, target_slice_size_meters)
        label_slices, _= slice_image(label_path, target_slice_size_meters)
        for ps_sl, p1_sl, lab_sl in zip(pshp_slices, p1bs_slices, label_slices):
            # ps_sl => shape (C=4, H, W)
            # p1_sl => shape (C=1, H, W)
            # lab_sl => shape (C=1, H, W)
            # Move axis => (H,W, C)
            ps_sl = np.moveaxis(ps_sl, 0, -1)  # => (H,W,4)
            p1_sl = np.moveaxis(p1_sl, 0, -1)  # => (H,W,1)
            x_sl = np.concatenate([ps_sl, p1_sl], axis=-1)  # => (H,W,5)
            x_sl = x_sl.astype('float32') / 255.0
            y_sl = np.moveaxis(lab_sl, 0, -1).astype('float32')  # => (H,W,1)

            yield x_sl, y_sl

##########################################################
# C) Building the DINOv2-like ViT + DPT decoder
##########################################################
class DPTHead(nn.Module):
    """
    A simplified DPT head:
    We take multi-scale features from the backbone, do the "DPT" upsampling.
    For a minimal example, we show a single-scale approach.
    If you want the full multi-scale skip connection DPT, see official code:
    https://github.com/isl-org/DPT
    """
    def __init__(self, in_channels=768, out_channels=1, up_factor=4):
        super().__init__()
        # A small example: ConvTransposed upsampling
        hidden = in_channels // 2
        self.conv_trans = nn.ConvTranspose2d(
            in_channels, hidden, kernel_size=up_factor, stride=up_factor
        )
        self.bn = nn.BatchNorm2d(hidden)
        self.act = nn.ReLU(inplace=True)
        self.out_conv = nn.Conv2d(hidden, out_channels, kernel_size=1)

    def forward(self, x):
        # x => shape (B, in_channels, H', W')
        x = self.conv_trans(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.out_conv(x)
        return x

class DinoV2DPTSegModel(nn.Module):
    """
    Combine:
      - DINOv2-based ViT backbone (Large or Huge) from local or from timm,
      - Minimal DPT Head for upsampling to ~4x
      - Possibly final upsample to match input.
    """
    def __init__(
        self,
        # backbone_name='vit_base_patch16_224',
        # For DINOv2: 'facebook/dinov2_vitl14' or 'dinov2_vith14' if using HuggingFace or custom
        backbone_name='vit_large_patch14_dinov2.lvd142m',
        out_channels=1,
        slice_3ch=True,
        pretrained_weights_path=None
    ):
        super().__init__()

        # 1) Load or create your DINOv2 backbone
        # If timm has it or you have your custom code
        # For demonstration, let's do a "timm" approach:
        self.backbone = timm.create_model(backbone_name, pretrained=False)
        # remove classifier head
        if hasattr(self.backbone, 'head'):
            self.backbone.head = nn.Identity()
        elif hasattr(self.backbone, 'fc'):
            self.backbone.fc = nn.Identity()

        # 2) If you have local pretrained weights from DINOv2:
        if pretrained_weights_path is not None:
            # e.g. safetensors official pretrained model [WHEN STARTING FOR FIRST TIME]
            sd_local = safetensors.torch.load_file(pretrained_weights_path, device='cpu')
            self.backbone.load_state_dict(sd_local, strict=False)

            # ## RESUME TRAINING FROM .PHT FILE [OPTION2]
            # sd_local = torch.load(pretrained_weights_path, map_location='cpu')
            # self.backbone.load_state_dict(sd_local, strict=True)

        # 3) we can keep a flag slice_3ch => if True, we'll slice x[:, :3]
        self.slice_3ch = slice_3ch

        # 4) figure out backbone channels
        if hasattr(self.backbone, 'num_features'):
            in_ch = self.backbone.num_features
        else:
            in_ch = 768  # or 1024, etc. for Large/Huge

        # 5) Build a DPT head
        self.dpt = DPTHead(in_channels=in_ch, out_channels=out_channels, up_factor=4)

    def forward(self, x):
        """
        x => shape (B, 5, H, W) if you have 5 channels
        or (B,3,H,W)
        We'll do:
          - optionally slice to first 3 channels,
          - maybe do a 224×224 or 518×518 type resize if required by DINO,
          - run backbone => shape (B, in_ch, H//16, W//16),
          - run DPT => shape (B, 1, something, something)
          - final upsample to match input size
        """
        B, C, H, W = x.shape

        if self.slice_3ch and C > 3:
            x = x[:, :3, :, :]  # shape (B,3,H,W)

        # Suppose the DINO was trained at 224×224 or 518×518. We'll do e.g. 224×224:
        # or if using a large patch => e.g. 14 => up to you.
        # For demonstration:
        x_resize = F.interpolate(x, size=(518,518), mode='bilinear', align_corners=False)

        # forward through the DINO backbone
        feats = self.backbone.forward_features(x_resize)
        # feats shape depends on the timm backbone.
        # Often => (B, n_patches, hidden_size) or (B, hidden_size, h', w')

        # let's handle the typical (B, n, c). We'll remove the CLS token
        if len(feats.shape) == 3:
            # e.g. (B, 197, 768) => remove CLS => (B,196,768)
            feats = feats[:, 1:, :]
            # reshape => e.g. 14×14
            n = feats.shape[1]
            h_ = int(n**0.5)
            feats = feats.transpose(1,2).reshape(B, -1, h_, h_)
        else:
            # if e.g. (B, c, h', w'), we might do feats = feats
            pass

        # run DPT
        out_dpt = self.dpt(feats)  # => e.g. (B,1, h'*4, w'*4)

        # final upsample to the original (H,W)
        out_final = F.interpolate(out_dpt, size=(H,W), mode='bilinear', align_corners=False)
        return out_final

##########################################################
# D) Training loop with early stopping
##########################################################
# def save_checkpoint(model, path="dino2_dpt_best.pth"):
#     torch.save(model.state_dict(), path)

def save_checkpoint(model, optimizer, path="dino2_dpt_best.pth"):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
        # "epoch": epoch
    }, path)


def load_checkpoint(model, path="dino2_dpt_best.pth", device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint)

def load_checkpoint2(model, path="dino2_dpt_best.pth", device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint)
    return model


# def load_checkpoint2(model, optimizer, path="dino2_dpt_best.pth", device="cpu"):
#     checkpoint = torch.load(path, map_location=device)
#
#     # 1) Load model
#     # model.load_state_dict(checkpoint["model_state_dict"])
#     model.load_state_dict(checkpoint, strict=True)
#
#     # 2) Load optimizer
#     # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
#     optimizer.load_state_dict(checkpoint)
#
#     # # 3) Resume epoch
#     # # For example, if you stored the epoch in your checkpoint:
#     # start_epoch = checkpoint["epoch"] + 1
#
#     return model, optimizer


def train_dino_dpt(
        model,
        optimizer,  ### ADDED LATER
        train_tf_dataset,
        val_tf_dataset,
        epochs=50,
        device='cpu',
        patience=10,
        ckpt_path="dino2_dpt_best.pth"
    ):
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    # optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss = float('inf')
    no_improve = 0

    for ep in range(epochs):
        ################ TRAIN ###############
        model.train()
        tr_loss, tr_steps = 0.0, 0
        for x_tf, y_tf in train_tf_dataset:
            x_np = x_tf.numpy()  # (B,H,W,5)
            y_np = y_tf.numpy()  # (B,H,W,1)

            x_np = np.moveaxis(x_np, -1, 1) # => (B,5,H,W)
            y_np = np.moveaxis(y_np, -1, 1) # => (B,1,H,W)

            x_t = torch.from_numpy(x_np).float().to(device)
            y_t = torch.from_numpy(y_np).float().to(device)

            optimizer.zero_grad()
            logits = model(x_t)  # => (B,1,H,W)

            loss = criterion(logits, y_t)
            loss.backward()
            optimizer.step()

            tr_loss += loss.item()
            tr_steps += 1

        avg_tr = tr_loss / max(1, tr_steps)

        ################ VAL ###############
        model.eval()
        v_loss, v_steps = 0.0, 0
        with torch.no_grad():
            for xv_tf, yv_tf in val_tf_dataset:
                xv_np = xv_tf.numpy()
                yv_np = yv_tf.numpy()

                xv_np = np.moveaxis(xv_np, -1, 1)
                yv_np = np.moveaxis(yv_np, -1, 1)

                xv_t = torch.from_numpy(xv_np).float().to(device)
                yv_t = torch.from_numpy(yv_np).float().to(device)

                val_logits = model(xv_t)
                vloss = criterion(val_logits, yv_t)
                v_loss += vloss.item()
                v_steps += 1

        avg_val = v_loss / max(1, v_steps)

        print(f"[Epoch {ep+1}/{epochs}] train={avg_tr:.4f}, val={avg_val:.4f}")

        # checkpoint if improved
        if avg_val < best_val_loss:
            print(f"   -> best val improved: {avg_val:.4f} (old {best_val_loss:.4f})")
            best_val_loss = avg_val
            no_improve = 0
            # save_checkpoint(model, ckpt_path)
            save_checkpoint(model, optimizer, ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping!")
                print(f"   -> reloading best from {ckpt_path}")
                # load_checkpoint(model, ckpt_path, device=device)
                model, optimizer = load_checkpoint(model, optimizer, ckpt_path, device=device)  # Now loads optimizer too
                break

    return model


### TRAIN

# pshp_dir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/x_train'
# p1sb_dir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/P1BS_files'
# y_dir = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/UNET_TRAINING_DATA_v2/y_train2_shrubs_p70_v2/final'
image_triplets = []

for fn in os.listdir(pshp_dir):
    if fn.endswith('.tif'):
        pshp_path = os.path.join(pshp_dir, fn)
        p1bs_path = pshp_path.replace("x_train","P1BS_files").replace("PSHP","P1BS")
        base_name = fn[:-4]
        label_path = None
        for yfn in os.listdir(y_dir):
            if yfn.startswith(base_name) and yfn.endswith('.tif'):
                label_path = os.path.join(y_dir, yfn)
                break
        if label_path:
            image_triplets.append((pshp_path, p1bs_path, label_path))

### 2) Train/Val Split
train_triplets, val_triplets = train_test_split(
    image_triplets, test_size=0.1, random_state=42
)

### 3) tf.data Datasets
batch_size = 8
train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator_from_triplets(train_triplets),
    output_signature=(
        tf.TensorSpec(shape=(None,None,5), dtype=tf.float32),
        tf.TensorSpec(shape=(None,None,1), dtype=tf.float32)
    )
).batch(batch_size).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator_from_triplets(val_triplets),
    output_signature=(
        tf.TensorSpec(shape=(None,None,5), dtype=tf.float32),
        tf.TensorSpec(shape=(None,None,1), dtype=tf.float32)
    )
).batch(batch_size).prefetch(tf.data.AUTOTUNE)



### -- -- -- FIRST TRAINING  -- -- -- ###
### 4) Build the Dino+DPT model
model_name = "vit_large_patch14_dinov2.lvd142m"  # or your DINOv2 large/huge
# local_weights_path = "/path/to/dinov2_vitl14.safetensors"  # optional
model_dinodpt = DinoV2DPTSegModel(
    backbone_name=model_name,
    out_channels=1,
    slice_3ch=True,
    pretrained_weights_path=local_weights_path
)

optimizer = optim.Adam(model_dinodpt.parameters(), lr=1e-4)

## --- --- ---  5) Train
best_model = train_dino_dpt(
    model=model_dinodpt,
    optimizer=optimizer,
    train_tf_dataset=train_dataset,
    val_tf_dataset=val_dataset,
    epochs=1000,
    device="cpu",
    patience=10,
    # ckpt_path="shrub_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT.pth"
    # ckpt_path="wettundra_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT.pth"
    ckpt_path="lakesrivers_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT.pth"
)







## --- --- --- RESUME TRAINING
model2 = DinoV2DPTSegModel(
    backbone_name="vit_large_patch14_dinov2.lvd142m",
    out_channels=1,
    slice_3ch=True,
    pretrained_weights_path=None  # We'll handle the load manually below
)

# Then proceed to define your optimizer and do your training loop ...
optimizer = optim.Adam(model2.parameters(), lr=1e-4)

# 3. Optionally create your LR scheduler or other training components if you have them
# scheduler = ...

# saved_weights_path = '/Users/radakovicd1/Dropbox Amsterdam Dropbox/Darko Radakovic/above_nasa/machinelearning_datasets/original_datasets_from_slice/ST_CNN_test/models/shrub_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_48h.pth'
# saved_weights_path = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/models/shrub_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_48h.pth'
# wettundra
saved_weights_path = '/Volumes/OWC Express 1M2/nasa_above/models/wettundra_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_ADAPT_v4.pth'
# sd_local = torch.load(saved_weights_path, map_location='cpu')
# model.load_state_dict(sd_local, strict=True)

model2 = load_checkpoint2(model2,path=saved_weights_path,device="cpu")
#
# model, optimizer = load_checkpoint(
#     model,
#     optimizer,
#     path=saved_weights_path,
#     device="cpu"
# )

# 5. Now continue training with your train loop (or your `train_dino_dpt` function).
best_model = train_dino_dpt(
    model=model2,
    optimizer=optimizer,
    train_tf_dataset=train_dataset,
    val_tf_dataset=val_dataset,
    epochs=1000,
    device="cpu",
    patience=10,
    # ckpt_path="shrub_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_v1_ADAPT_with48h.pth"
    ckpt_path="wettundra_VIT14_DINOv2pretrained_trainingdata3_RADnor_400px_PSHP_P1BS_ADAPT_v4_48h_resumed.pth"
    # You could also pass in an initial epoch to skip the ones you've done
)







### 6) Inference Example
# local_weights_path = '/explore/nobackup/people/dradako1/jupyternobackup/cnn_darko/UNET_TRAINING_DATA_v2/models/VIT_pretrained_models--timm--vit_large_patch14_dinov2.lvd142m/snapshots/2e99f078e7011d363e5e6647c7b0383e11788b68/model.safetensors'

# best_model.eval()
# Suppose you have a single tile => shape (C,H,W) => (5,H,W).
# Then do:
#   tile_t = (1,5,H,W) => best_model(tile_t)
#   => (1,1,H,W). Then do sigmoid => threshold => ...







