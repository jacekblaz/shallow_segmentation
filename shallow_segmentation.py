import matplotlib.pyplot as plt
import numpy as np
import cv2

from skimage.data import astronaut
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from sklearn.cluster import KMeans
from skimage.color import rgb2gray
from skimage.filters import sobel


# function takes mask(another notation: segment, regions, superpixels, filter, objects curves) and original image as input
# mask is numpy array in which every cell value indicades what is corresponding pixel's segment
# (to which superpixel its belonges to) with int as label ([00001] <- mask for 3 labels on pic 3x5
#                                                          [00012]
#                                                          [30012])
def color_regions(segments, img):

    #take into consideration just one superpixel(segment, region etc.)
    for current_segment in range(0, len(segments)):
        avg = [0, 0, 0]
        #find all pixels in superpixel
        pixels_in_segment = np.argwhere(segments == current_segment)

        #cannot divide by zero later, idk how to fix it better- some condition should be applied
        if not pixels_in_segment.any():
            break

        #go through every pixel in superpixel to compute avarage colour[3 channels for rgb]
        for pixel in pixels_in_segment:
            avg[0] += img[pixel[0], pixel[1], 0]
            avg[1] += img[pixel[0], pixel[1], 1]
            avg[2] += img[pixel[0], pixel[1], 2]
        avg = [x / len(pixels_in_segment) for x in avg]

        #apply avarage color to every pixel in superpixel
        for pixel in pixels_in_segment:
            img[pixel[0], pixel[1], 0] = avg[0]
            img[pixel[0], pixel[1], 1] = avg[1]
            img[pixel[0], pixel[1], 2] = avg[2]

    return img


def kmeans(img, n_clusters):
    original_shape = img.shape
    img = img.reshape((img.shape[0] * img.shape[1], 3))
    labeled_img = KMeans(n_clusters).fit_predict(img)
    labeled_img = np.asarray(labeled_img)
    labeled_img = labeled_img.reshape((original_shape[0], original_shape[1]))
    return labeled_img



def show_results(n_clusters):
    print("Felzenszwalb number of segments: {}".format(len(np.unique(segments_fz))))

    fig, ax = plt.subplots(3, 2, figsize=(30, 30), sharex=True, sharey=True)
    ax[0, 0].imshow(raw_img)
    ax[0, 0].set_title("Original pic")
    ax[0, 1].imshow(kmeans(raw_img, n_clusters=n_clusters))
    ax[0, 1].set_title("Only kmeans")
    ax[1, 0].imshow(mark_boundaries(raw_img, segments_fz))
    ax[1, 0].set_title("Felzenszwalbs's method for curves")
    ax[1, 1].imshow(color_regions(segments_fz, raw_img))
    ax[1, 1].set_title("Avarage coloring on regions from Felzenszwalbs's")
    ax[2, 0].imshow(mark_boundaries(color_regions(segments_fz, raw_img), segments_fz))
    ax[2, 0].set_title("Felzenszwalbs's method and avarage coloring")
    ax[2, 1].imshow(kmeans(color_regions(segments_fz, raw_img), n_clusters=n_clusters))
    ax[2, 1].set_title("k-means on avg colors")
    for a in ax.ravel():
        a.set_axis_off()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    raw_img = cv2.imread('sample_pic.jpg')
    raw_img = img_as_float(raw_img[::2, ::2])
    #tweak scale, sigma, min_size, n_clusters to achieve better results
    #http://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.felzenszwalb
    segments_fz = felzenszwalb(raw_img, scale=90, sigma=0.8, min_size=15)
    show_results(n_clusters=3)