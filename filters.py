import math
import os
import pickle
import timeit
import numba
from numba import cuda
from numba import njit, prange
import imageio
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d


def correlation_gpu(kernel, image):
    '''Correlate using gpu
    Parameters
    ----------
    kernel : numpy array
        A small matrix
    image : numpy array
        A larger matrix of the image pixels

    Return
    ------
    An numpy array of same shape as image
    '''
    global m
    global n

    m = kernel.shape[0]
    n = kernel.shape[1]

    mpad = int(m / 2)
    npad = int(n / 2)

    blocks_in_grid = (math.ceil(image.shape[0] / 28), math.ceil(image.shape[1] / 28))

    image = np.pad(image, ((mpad, mpad), (npad, npad)))

    kernel_cuda = cuda.to_device(kernel)
    image_cuda = cuda.to_device(image)
    output_cuda = cuda.device_array((image.shape[0], image.shape[1]), dtype=image.dtype)

    correlation_cuda_kernel[blocks_in_grid, (28, 28)](kernel_cuda, image_cuda, output_cuda)

    output = output_cuda.copy_to_host()

    return output


@cuda.jit
def correlation_cuda_kernel(kernel, image, output):
    i, j = cuda.grid(2)

    tidx = cuda.threadIdx.x
    tidy = cuda.threadIdx.y

    shared_kernel_arr = cuda.shared.array(shape=(m, n), dtype=numba.float64)

    if tidx < m and tidy < n:
        shared_kernel_arr[tidx, tidy] = kernel[tidx, tidy]

    cuda.syncthreads()

    shared_image_arr = cuda.shared.array(shape=(28, 28), dtype=numba.float64)

    mpad = int(m / 2)
    npad = int(n / 2)   

    shared_image_arr[tidx, tidy] = image[i + mpad, j + npad]

    cuda.syncthreads()

    if i >= output.shape[0] or j >= output.shape[1]:
        return

    for x in range(-(m // 2), (m + 1) // 2):
        for y in range(-(n // 2), (n + 1) // 2):
            if tidx + x > 0 and tidx + x < shared_image_arr.shape[0] and tidy + y > 0 and tidy + y < \
                    shared_image_arr.shape[1]:
                current_val = shared_kernel_arr[int(x + math.floor(m / 2)), int(y + math.floor(n / 2))]
                current_val *= shared_image_arr[int(tidx + x), int(tidy + y)]
                output[i, j] += current_val
            else:
                current_val = shared_kernel_arr[int(x + math.floor(m / 2)), int(y + math.floor(n / 2))]
                current_val *= image[int(i + x) + mpad, int(j + y) + npad]
                output[i, j] += current_val


@njit
def correlation_numba(kernel, image):
    '''Correlate using numba
    Parameters
    ----------
    kernel : numpy array
        A small matrix
    image : numpy array
        A larger matrix of the image pixels

    Return
    ------
    An numpy array of same shape as image
    '''

    kernel_rows, kernel_cols = kernel.shape
    image_rows, image_cols = image.shape

    # Calculate the padding required for valid correlation
    pad_rows = kernel_rows // 2
    pad_cols = kernel_cols // 2

    # Initialize the result matrix with zeros
    result = np.zeros_like(image)

    # Perform correlation operation
    for i in prange(pad_rows, image_rows - pad_rows):
        for j in prange(pad_cols, image_cols - pad_cols):
            # Compute the correlation at the current position
            sum = 0.0
            for m in prange(kernel_rows):
                for n in prange(kernel_cols):
                    sum += kernel[m, n] * image[i - pad_rows + m, j - pad_cols + n]
            result[i, j] = sum

    return result


def sobel_operator():
    '''Load the image and perform the operator
        ----------
        Return
        ------
        An numpy array of the image
        '''
    pic = load_image()
    values = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
    sobel_filter = np.array(values)
    G_x = correlation_numba(sobel_filter, pic)
    G_y = correlation_numba(np.transpose(sobel_filter), pic)


    result = np.zeros((G_x.shape[0], G_x.shape[1]))
    for i in range(int(G_x.shape[0])):
        for j in range(int(G_x.shape[1])):
            result[i, j] = np.sqrt(((G_x[i, j] ** 2) + (G_y[i, j] ** 2)))

    return result

def sobel_kernel_first():
    kernel = [[3, 0, -3], [10, 0, -10], [3, 0, -3]]
    return correlation_numba(np.array(kernel), load_image())

def sobel_kernel_second():
    kernel = np.array([[1, 0, -1], [2, 0, -1], [1, 0, -2], [2, 0 , -2], [1, 0, -1]])
    return correlation_numba(kernel, load_image())

def sobel_kernel_third():
    kernel = np.array([[1, 1, -1], [1, 0, 1], [1, 1, 1]])
    return correlation_numba(kernel, load_image())

# def sobel_operator_cpu_conv():
#     pic = load_image()
#     values = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
#     sobel_filter = np.array(values)
#     G_x = convolve2d(pic, sobel_filter, mode='same')
#     G_y = convolve2d(pic, np.transpose(sobel_filter), mode='same')
#
#     result = np.zeros((G_x.shape[0], G_x.shape[1]))
#     for i in range(int(G_x.shape[0])):
#         for j in range(int(G_x.shape[1])):
#             result[i, j] = np.sqrt(((G_x[i, j] ** 2) + (G_y[i, j] ** 2)))
#
#     return result

def load_image():
    fname = 'data/image.jpg'
    pic = imageio.imread(fname)
    to_gray = lambda rgb: np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    gray_pic = to_gray(pic)
    return gray_pic


def show_image(image):
    """ Plot an image with matplotlib

    Parameters
    ----------
    image: list
        2d list of pixels
    """
    plt.imshow(image, cmap='gray')
    plt.show()