import sys

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np
import yaml
from astropy.convolution import Box2DKernel, convolve
from osgeo import gdal
from scipy import stats


def fits2tiff(filename):
    hdulist = pyfits.open(filename)
    other = np.asarray(hdulist[0].data)
    earth = other.copy()
    earth = np.float32(earth)
    nrows, ncols = earth.shape[0], earth.shape[1]
    filename = filename[:-5] + ".tiff"
    dst_ds = gdal.GetDriverByName("GTiff").Create(
        filename, ncols, nrows, 1, gdal.GDT_Float32
    )
    dst_ds.SetProjection("epsg:4326")
    dst_ds.SetGeoTransform(
        (
            hdulist[0].header["CRVAL1"],
            hdulist[0].header["CRPIX1"],
            0,
            hdulist[0].header["CRVAL2"],
            0,
            hdulist[0].header["CRPIX2"],
        )
    )
    dst_ds.GetRasterBand(1).WriteArray(earth)
    return filename


def open_tiff(filename, dtype=np.float32):
    if filename.endswith("fits"):
        filename = fits2tiff(filename)

    # Load file, and access the band and get a NumPy array
    src = gdal.Open(filename, gdal.GA_Update)
    band = src.GetRasterBand(1)
    ar = band.ReadAsArray()
    return src, ar


with open("params") as f:
    p = yaml.safe_load(f)
basename = p["basename"]

# open intensity and technology (spectral class) images
src, image_intensity = open_tiff(f"popular/Corr_{basename}_ImpactVlG_GR.fits")
src, image_tech = open_tiff(f"popular/Corr_{basename}CompositeW.fits")

# open signal to noise ratio images
images_snr = [
    open_tiff(f"quality/Corr_{basename}SNR{band}o2_rect.tiff")[1]
    for band in ["R1", "G2", "G3", "B4"]
]

# 0. Find saturated pixels that have no value in the intensity image and
#    replace the nan by the maximum of the image
sat = sys.float_info.max
novalue = -1e30
# changing sat values (nan) to number (sat value), the nan that corresponds
# to no data will change to 0
images_snr = np.nan_to_num(images_snr)
# changing nan to very small number to be able to find them
image_intensity[
    (np.isnan(image_intensity) | (image_intensity == 0))
    & (images_snr == sat).any(axis=0)
] = np.nanmax(image_intensity)

# 1. Start of treatment : elimination of noise
# 1a. Eliminate negative data, mistake resulting of the pre-treatment
image_intensity[image_intensity < 0] = np.nan


# 1b. Statistical data
def compute_stats(image):
    mean = np.nanmean(image)
    median = np.nanmedian(image)
    standard_deviation = np.nanstd(image)
    mode = stats.mode(image[~np.isnan(image)], axis=None)[0][0]
    print(f"{mode=} {mean=} {median=} {standard_deviation=}")
    return mean, median, standard_deviation, mode


mean, median, standard_deviation, mode = compute_stats(image_intensity)

# 1c. We create a temporary map with the pixels of less value since the noise
#     should be smaller or around the same value as the less valuable pixels.
#     We then extract statistical data of these small values

small_values = image_intensity[image_intensity < median]
# mean2, median2, standard_deviation2, mode2 = compute_stats(small_values)

n, bins, patches = plt.hist(
    small_values[~np.isnan(small_values)].flatten(), bins=50
)
plt.show()
# index_max = np.argmax(n)
maxi = 0
for i, value in enumerate(n):
    if value < maxi:
        index = i
        break
    else:
        maxi = value

# 1d. Using the statistical data on hand, we estimate the value of the
# background, defined as the center value of the bin with most pixels.
# background= (bins[index_max]+bins[index_max+1])/2 + standard_deviation2*3
background = bins[index + 1]  # the right hand border of the bin

plt.figure()
plt.imshow(image_intensity, cmap="rainbow")
plt.colorbar()
plt.title("Intensity with noise")
plt.show()

# this value can change. Modifie accordingly in order to achieve the best
#    results (see step 1.f)
image_intensity -= background

# 1e. Eliminating negative pixels created by our treatment of the noise
#     in the image
image_intensity[image_intensity < 0] = np.nan
small_values = image_intensity[image_intensity < median]
n, bins, patches = plt.hist(
    small_values[~np.isnan(small_values)].flatten(), bins=100
)
plt.show()

# 1f. Compare your image to the ones in ReadMe for reference as well as Google
#     maps to know which zones should emit light or not. We want to eliminate a
#     certain quantity of small value pixels in aeras that shouldn't emit light
#     At the same time, we want to limit the values eliminated in aeras where
#     light should be emitted

plt.figure()
plt.imshow(image_intensity, cmap="rainbow")
plt.colorbar()
plt.title("Intensity without noise")
plt.show()

# 1.5. Finding unexplicable (so far) void pixels surrounded by high intensity
#      pixels and filling them with the mean of the pixels around
#      creating binary images in intensity, value=1, nan=0


def image_to_binary(image):
    im = image.copy()
    im[im >= 0] = 1
    im[np.isnan(image)] = 0
    return im


def binary_mode_classes(im_tech, window):
    tech_conv = [
        convolve(im_tech == i, Box2DKernel(width=window)) for i in range(1, 7)
    ]
    im_t = np.argmax(tech_conv, axis=0) + 1
    im_t[np.prod(tech_conv, axis=0) == 0] = 0
    return im_t


def convolution_nb_void(image, im_tech, window, keep_value):
    im = image.copy()
    nb_nan_binary = convolve(image_to_binary(image), Box2DKernel(width=window))
    nb_nan_real = convolve(np.nan_to_num(image), Box2DKernel(width=window))
    mean = np.nanmean(image)
    nb_nan_binary_without0 = nb_nan_binary.copy()
    nb_nan_binary_without0[nb_nan_binary_without0 == 0] = 1
    mask = (
        (np.nan_to_num(image) == 0)
        & (nb_nan_binary > keep_value / window ** 2)
        & ((nb_nan_real / nb_nan_binary_without0) > mean)
    )
    tech = im_tech.copy()
    im_t = binary_mode_classes(im_tech, window)
    im[mask] = nb_nan_real[mask] / nb_nan_binary[mask]
    tech[mask] = im_t[mask]

    return im, tech


image_tech[image_tech == 0] = np.nan
image_intensity, image_tech = convolution_nb_void(
    image_intensity, image_tech, window=3, keep_value=5
)


# 2. Elimination of remaining values in dark aeras
#    If your image is already free of valued pixels in dark aeras following
#    step 1, comment this section and skip to step 3

# 2a. We define our convolution fonction
def convolution_nb_nan(image, window, keep_value):
    im = image.copy()
    nb_nan = convolve(image_to_binary(image), Box2DKernel(width=window))
    im[
        (np.nan_to_num(im) != 0) & ((window ** 2 * nb_nan) < keep_value)
    ] = np.nan
    return im  # np.nan_to_num(im)


# 2b. We create a binary image where value pixels are equal to 1 and NaN pixels
#     are equal to 0
image_intensity = convolution_nb_nan(image_intensity, window=4, keep_value=2)

# 2c. Second convolution if necessary. If not, skip to step 3
image_intensity = convolution_nb_nan(image_intensity, window=4, keep_value=2)

# 2d. Compare your image to the examples in the ReadMe
plt.figure()
plt.imshow(image_intensity, cmap="rainbow")
plt.colorbar()
plt.title("Intensity with clean dark aeras")
plt.show()


# 3. Concordance between intensity and technology images


def int_tech_comparison(intensity, im_tech):
    tech = im_tech.copy()
    im_t = binary_mode_classes(im_tech, window=3)
    tech[((np.nan_to_num(intensity) == 0) & (np.nan_to_num(im_tech) != 0))] = 0
    mask = (np.nan_to_num(intensity) != 0) & (np.nan_to_num(im_tech) == 0)
    tech[mask] = im_t[mask]
    return tech


image_tech = int_tech_comparison(image_intensity, image_tech)
image_tech[image_tech == 0] = np.nan

# plt.figure()
# plt.imshow(image_intensity, cmap="rainbow")
# plt.colorbar()
# plt.title("Image Intensity")
# plt.show()
#
# plt.figure()
# plt.imshow(image_tech, cmap="rainbow")
# plt.colorbar()
# plt.title("Technology")
# plt.show()


# 7. Saving results


def save_geotiff(filename, data):
    nband = 1
    nrow, ncol = data.shape
    driver = gdal.GetDriverByName("GTiff")
    dst_dataset = driver.Create(
        filename + ".tiff", ncol, nrow, nband, gdal.GDT_Float32
    )
    # sets same geotransform as input
    dst_dataset.SetGeoTransform(src.GetGeoTransform())
    # sets same projection as input
    dst_dataset.SetProjection(src.GetProjection())
    dst_dataset.GetRasterBand(1).WriteArray(data.astype(float))
    dst_dataset = None


# 7a. Saving intensity

save_geotiff("Image_Vrad", image_intensity)
np.save("Image_Vrad", image_intensity)

# 7b. Saving MSI
# save_geotiff('Image_MSI',Blur_MSI)
# np.save('Image_MSI',Blur_MSI)

# 7c. Saving Impact MSI
# save_geotiff('Impact_MSI',Impact_MSI)
# np.save('Impact_MSI',Impact_MSI)

# 7d. Saving Technology
save_geotiff("tech_image", image_tech)
np.save("tech_image", image_tech)
