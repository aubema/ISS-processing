import numpy as np
import matplotlib.pyplot as plt
import sys
from astropy.io import fits
from astropy.utils.data import download_file
from scipy import stats
from astropy.convolution import convolve, Gaussian2DKernel, Box2DKernel
from osgeo import gdal

basename = "./Canada_cities/Calgary_2021/Corr_iss030e066406"

def open_tiff(filename,dtype=np.float32):
	# Load file, and access the band and get a NumPy array
	src = gdal.Open(filename, gdal.GA_Update)
	band = src.GetRasterBand(1)
	ar = band.ReadAsArray()
	return src, ar

#open intensity and technology (spectral class) images
src, image_intensity = open_tiff(basename+'_ImpactVlG_GR.tiff') #Impact
src, image_tech = open_tiff(basename + 'Composite.tiff')


#open signal to noise ratio images
src, image_snrB = open_tiff(basename + 'SNRB4o2_rect.tiff')
src, image_snrR = open_tiff(basename + 'SNRR1o2_rect.tiff')
src, image_snrG2 = open_tiff(basename + 'SNRG2o2_rect.tiff')
src, image_snrG3 = open_tiff(basename + 'SNRG3o2_rect.tiff')

##0. Find saturated pixels that have no value in the intensity image and replace the nan by the maximum of the image
sat=1.7976931348623157e+308
novalue= -1e+30
image_snrB = np.nan_to_num(image_snrB) #changing sat values (nan) to number (sat value), the nan that corresponds to no data will change to 0
image_snrR = np.nan_to_num(image_snrR)
image_snrG2 = np.nan_to_num(image_snrG2)
image_snrG3 = np.nan_to_num(image_snrG3)
image_int = np.nan_to_num(image_intensity, nan=novalue) #changing nan to very small number to be able to find them
image_intensity[((image_int==novalue) | (image_int==0)) & ((image_snrB==sat) | (image_snrR==sat) | (image_snrG2==sat) | (image_snrG3==sat))] = np.nanmax(image_intensity)

## 1. Start of treatment : elimination of noise
## 1a. Eliminate negative data, as these are a mistake resulting of the pre-treatment
image_intensity[image_intensity<0] = np.nan


## 1b. Statistical data
def compute_stats(image):
	mean = np.nanmean(image)
	median = np.nanmedian(image)
	standard_deviation = np.nanstd(image)
	mode = stats.mode(image[~np.isnan(image)],axis=None)[0][0]
	print(f"{mode:} {mean:} {median:} {standard_deviation:}")
	return mean, median, standard_deviation, mode

mean, median, standard_deviation, mode = compute_stats(image_intensity)

## 1c. We create a temporary map with the pixels of less value since the noise should be
##     smaller or around the same value as the less valuable pixels.
##     We then extract statistical data of these small values

small_values = image_intensity[image_intensity<median]
#mean2, median2, standard_deviation2, mode2 = compute_stats(small_values)

(n, bins, patches) = plt.hist(small_values[~np.isnan(small_values)].flatten(),bins=50)
plt.show()
#index_max = np.argmax(n)
maxi=0
for i,value in enumerate(n):
        if value<maxi:
                index=i
                break
        else:
                maxi=value

## 1d. Using the statistical data on hand, we estimate the value of the background, defined as the center value of the bin with most pixels.
#background= (bins[index_max]+bins[index_max+1])/2 + standard_deviation2*3
background=bins[index+1] #the right hand border of the bin

plt.figure()
plt.imshow(image_intensity, cmap="rainbow")
plt.colorbar()
plt.title('Intensity with noise')
plt.show()

## this value can change. Modifie accordingly in order to achieve the best results (see step 1.f)
image_intensity -= background

## 1e. Eliminating negative pixels created by our treatment of the noise in the image
image_intensity[image_intensity<0] = np.nan
small_values = image_intensity[image_intensity<median]
(n, bins, patches) = plt.hist(small_values[~np.isnan(small_values)].flatten(),bins=100)
plt.show()

## 1f. Compare your image to the ones in ReadMe for reference as well as Google maps to know which zones should emit light or not
##     We want to eliminate a certain quantity of small value pixels in aeras that shouldn't emit light
##     At the same time, we want to limit the values eliminated in aeras where light should be emitted

plt.figure()
plt.imshow(image_intensity, cmap="rainbow")
plt.colorbar()
plt.title('Intensity without noise')
plt.show()

#1.5. Finding unexplicable (so far) void pixels surrounded by high intensity pixels and filling them with the mean of the pixels around
def image_to_binary(image): #creating binary images in intensity, value=1, nan=0
	im = image.copy()
	im[im>=0] = 1
	im[np.isnan(image)] = 0
	return im
def tech_to_binary(image,tech): #creating binary images for a tech class, class=1, dif_class=0
        im= image.copy()
        im[im!=tech] = 0
        im[im==tech] = 1
        return im
def binary_mode_classes(im_tech, window):
        im_t=im_tech.copy()
        tech_1_conv=convolve(tech_to_binary(im_t,1), Box2DKernel(width=window))
        tech_2_conv=convolve(tech_to_binary(im_t,2), Box2DKernel(width=window))
        tech_3_conv=convolve(tech_to_binary(im_t,3), Box2DKernel(width=window))
        tech_4_conv=convolve(tech_to_binary(im_t,4), Box2DKernel(width=window))
        tech_5_conv=convolve(tech_to_binary(im_t,5), Box2DKernel(width=window))
        tech_6_conv=convolve(tech_to_binary(im_t,6), Box2DKernel(width=window))
        im_t[(tech_1_conv==0) & (tech_2_conv==0) & (tech_3_conv==0) & (tech_4_conv==0) & (tech_5_conv==0) & (tech_6_conv==0)]=0
        im_t[(im_t!=0) & (tech_1_conv>=tech_2_conv) & (tech_1_conv>=tech_3_conv) & (tech_1_conv>=tech_4_conv) & (tech_1_conv>=tech_5_conv) & (tech_1_conv>=tech_6_conv)]=1
        im_t[(im_t!=0) & (tech_2_conv>tech_1_conv) & (tech_2_conv>=tech_3_conv) & (tech_2_conv>=tech_4_conv) & (tech_2_conv>=tech_5_conv) & (tech_2_conv>=tech_6_conv)]=2
        im_t[(im_t!=0) & (tech_3_conv>tech_1_conv) & (tech_3_conv>tech_2_conv) & (tech_3_conv>=tech_4_conv) & (tech_3_conv>=tech_5_conv) & (tech_3_conv>=tech_6_conv)]=3
        im_t[(im_t!=0) & (tech_4_conv>tech_1_conv) & (tech_4_conv>tech_2_conv) & (tech_4_conv>tech_3_conv) & (tech_4_conv>=tech_5_conv) & (tech_4_conv>=tech_6_conv)]=4
        im_t[(im_t!=0) & (tech_5_conv>tech_1_conv) & (tech_5_conv>tech_2_conv) & (tech_5_conv>tech_3_conv) & (tech_5_conv>tech_4_conv) & (tech_5_conv>=tech_6_conv)]=5
        im_t[(im_t!=0) & (tech_6_conv>tech_1_conv) & (tech_6_conv>tech_2_conv) & (tech_6_conv>tech_3_conv) & (tech_6_conv>tech_4_conv) & (tech_6_conv>tech_5_conv)]=6
        return im_t

def convolution_nb_void(image, im_tech, window, keep_value):
	im = image.copy()
	nb_nan_binary = convolve(image_to_binary(image), Box2DKernel(width=window))
	nb_nan_real = convolve(np.nan_to_num(image), Box2DKernel(width=window))
	mean = np.nanmean(image)
	nb_nan_binary_without0=nb_nan_binary.copy()
	nb_nan_binary_without0[nb_nan_binary_without0==0]=1
	indices=np.argwhere((np.nan_to_num(image)==0) & (nb_nan_binary > keep_value/window**2) & ((nb_nan_real/nb_nan_binary_without0) > mean))
	tech=im_tech.copy()
	im_t=binary_mode_classes(im_tech, window)
	for i in indices:
                im[i[0]][i[1]] = nb_nan_real[i[0]][i[1]]/nb_nan_binary[i[0]][i[1]]
                tech[i[0]][i[1]] = im_t[i[0]][i[1]]
        
	return im, tech

image_tech[image_tech==0]=np.nan
image_intensity, image_tech = convolution_nb_void(image_intensity, image_tech, window=3, keep_value=5)


## 2. Elimination of remaining values in dark aeras
##    If your image is already free of valued pixels in dark aeras following step 1,
##    comment this section and skip to step 3

## 2a. We define our convolution fonction
def convolution_nb_nan(image, window, keep_value):
	im = image.copy()
	nb_nan = convolve(image_to_binary(image), Box2DKernel(width=window))
	im[(np.nan_to_num(im)!=0) & ((window**2 * nb_nan) < keep_value)] = np.nan
	return im #np.nan_to_num(im)

## 2b. We create a binary image where value pixels are equal to 1 and NaN pixels are equal to 0
image_intensity = convolution_nb_nan( image_intensity, window=4, keep_value=2 )

## 2c. Second convolution if necessary. If not, skip to step 3
image_intensity = convolution_nb_nan( image_intensity, window=4, keep_value=2 )

## 2d. Compare your image to the examples in the ReadMe
plt.figure()
plt.imshow(image_intensity, cmap="rainbow")
plt.colorbar()
plt.title('Intensity with clean dark aeras')
plt.show()


## 3. Concordance between intensity and technology images

def int_tech_comparison(intensity, im_tech):
        im_i=intensity.copy()
        tech=im_tech.copy()
        im_t=binary_mode_classes(im_tech,window=3)
        tech[((np.nan_to_num(im_i)==0) & (np.nan_to_num(im_tech)!=0))]=0
        indices=np.argwhere((np.nan_to_num(im_i)!=0) & (np.nan_to_num(im_tech)==0))
        for i in indices:
                tech[i[0]][i[1]] = im_t[i[0]][i[1]]
        return tech

image_tech = int_tech_comparison(image_intensity, image_tech)
image_tech[image_tech==0]=np.nan
'''
## 4. Creation of MSI map

MSI_array = [ 0.62, 0.43, 0.35, 0.08, 0.118, 0.017 ]
image_MSI = np.zeros_like(image_tech)
for i, msi in enumerate(MSI_array, 1):
	image_MSI[image_tech == i] = msi


## 5. Bluring

## 5a. Defining standard deviation
##     All measurements are in meters
pix_size = 8e-6 # pixel size
focal_distance = 400e-3 # focal distance
ISS_altitude = 400e3 # altutide of ISS
distance_dev = 25 # influence standard deviation of a single lamp
deviation = (distance_dev * focal_distance)/(pix_size*ISS_altitude) # influence standard deviation in pixel

## 5b. Defining the bluring function
def Convolution_without_zero(image,stdev=deviation):
	im = image.copy()
	mask = im==0
	im[mask] = np.nan
	blurred = convolve(im, Gaussian2DKernel(x_stddev=stdev))
	blurred[mask] = np.nan
	return blurred

## 5c. Bluring intensity image
Blur_intensity = Convolution_without_zero(image_intensity)

## 5d. Bluring MSI image
Blur_MSI = Convolution_without_zero(image_MSI)

## 6. Creation of impact MSI image
Impact_MSI = Blur_MSI*Blur_intensity

# Uncomment to show final images
plt.figure()
plt.imshow(Blur_MSI, cmap='rainbow')
plt.colorbar()
plt.title('Blured MSI')

plt.figure()
plt.imshow(Impact_MSI, cmap="rainbow")
plt.colorbar()
plt.title('Impact MSI')

plt.figure()
plt.imshow(Blur_intensity, cmap="rainbow")
plt.colorbar()
plt.title("Blured Intensity")
plt.show()
'''
plt.figure()
plt.imshow(image_intensity, cmap="rainbow")
plt.colorbar()
plt.title("Image Intensity")
plt.show()

plt.figure()
plt.imshow(image_tech, cmap="rainbow")
plt.colorbar()
plt.title("Technology")
plt.show()

## 7. Saving results
def save_geotiff( filename, data ):
	nband = 1
	nrow, ncol = data.shape
	driver = gdal.GetDriverByName('GTiff')
	dst_dataset = driver.Create(filename+".tiff", ncol, nrow, nband, gdal.GDT_Float32)
	dst_dataset.SetGeoTransform(src.GetGeoTransform())  ##sets same geotransform as input
	dst_dataset.SetProjection(src.GetProjection())  ##sets same projection as input
	dst_dataset.GetRasterBand(1).WriteArray(data.astype(float))
	dst_dataset = None


## 7a. Saving intensity

save_geotiff('Image_Vrad',image_intensity)
np.save('Image_Vrad',image_intensity)

## 7b. Saving MSI
#save_geotiff('Image_MSI',Blur_MSI)
#np.save('Image_MSI',Blur_MSI)

## 7c. Saving Impact MSI
#save_geotiff('Impact_MSI',Impact_MSI)
#np.save('Impact_MSI',Impact_MSI)

## 7d. Saving Technology
save_geotiff('tech_image',image_tech)
np.save('tech_image',image_tech)
