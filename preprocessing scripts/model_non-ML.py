
## test 2025 jan


import numpy as np
from osgeo import gdal, gdal_array
from skimage import morphology, util
from skimage.feature import graycomatrix, graycoprops
from scipy import ndimage


def detect_shrubs(input_path, output_path, ndvi_thresh=0.5, entropy_thresh=5.0):
    # Load pansharpened 4-band image (Blue, Green, Red, NIR)
    ds = gdal.Open(input_path)
    nir = ds.GetRasterBand(4).ReadAsArray().astype(np.float32)
    red = ds.GetRasterBand(3).ReadAsArray().astype(np.float32)

    # Compute NDVI (if not already calculated)
    ndvi = (nir - red) / (nir + red + 1e-12)

    # Texture analysis using GLCM entropy on NDVI
    def calculate_entropy(image):
        # Quantize image to 8-bit for GLCM
        image = util.img_as_ubyte((image + 1) / 2)  # Scale NDVI (-1 to 1) to 0-255

        # Create sliding window view
        window_size = 5
        windows = util.view_as_windows(image, (window_size, window_size))

        # Calculate entropy for each window
        entropy = np.zeros_like(image, dtype=np.float32)
        rows, cols = windows.shape[:2]

        for i in range(rows):
            for j in range(cols):
                window = windows[i, j]
                glcm = graycomatrix(window, distances=[1], angles=[0], levels=256,
                                    symmetric=True, normed=True)
                entropy[i, j] = graycoprops(glcm, 'entropy')[0, 0]

        return entropy

    # Calculate texture entropy
    entropy = calculate_entropy(ndvi)

    # Thresholding
    shrub_mask = (ndvi > ndvi_thresh) & (entropy > entropy_thresh)

    # Post-processing
    # 1. Morphological closing to fill gaps
    shrub_mask = ndimage.binary_closing(shrub_mask, structure=np.ones((3, 3)))

    # 2. Remove small objects
    shrub_mask = morphology.remove_small_objects(shrub_mask, min_size=50)

    # 3. Opening to remove noise
    shrub_mask = ndimage.binary_opening(shrub_mask, structure=np.ones((3, 3)))

    # # Save output
    # driver = gdal.GetDriverByName('GTiff')
    # out_ds = driver.Create(output_path, ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Byte)
    # out_ds.SetGeoTransform(ds.GetGeoTransform())
    # out_ds.SetProjection(ds.GetProjection())
    # out_ds.GetRasterBand(1).WriteArray(shrub_mask.astype(np.uint8) * 255)
    # out_ds.FlushCache()

    ds = None
    out_ds = None


# Example usage
detect_shrubs(
        input_path='/Volumes/OWC Express 1M2/clip_georef/ms6_area2_100m_clip_1950m_georef/x_train/WV03_20170811223418_1040010031305A00_PSHP_P001_NT_2000m_area2_GEOREF_clip1950m.tif',
        output_path="shrub_mask.tif",
        ndvi_thresh=0.5,  # Adjust based on your environment
        entropy_thresh=5.0  # Adjust based on texture analysis
    )



