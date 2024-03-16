#
#   @date:  [TODO: Today's date]
#   @author: [TODO: Student Names]
#
# This file is for the solutions of the wet part of HW2 in
# "Concurrent and Distributed Programming for Data processing
# and Machine Learning" course (02360370), Winter 2024
#
import pickle
from numba import cuda
import numba
from numba import njit, prange
import math
import imageio
import matplotlib.pyplot as plt
import numpy as np

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
    # Get the dimensions of the kernel and the image
    M, N = image.shape
    m, n = kernel.shape

    # Calculate the padding required for valid correlation
    mpad = m // 2
    npad = n // 2

    # Define the thread block dimensions
    threads_per_block = (32, 32)  # Adjust according to your GPU architecture

    # Calculate the grid dimensions
    blocks_x_axis = math.ceil(M / threads_per_block[0])
    blocks_y_axis = math.ceil(N / threads_per_block[1])
    blocks = (blocks_x_axis, blocks_y_axis)

    # Pad the image
    image_padded = np.pad(image, ((mpad, mpad), (npad, npad)), mode='constant')

    # Allocate device memory
    kernel_cuda = cuda.to_device(kernel)
    image_cuda = cuda.to_device(image_padded)
    output_cuda = cuda.device_array((M, N), dtype=np.float32)

    # Launch the CUDA kernel
    correlation_cuda_kernel[blocks, threads_per_block](kernel_cuda, image_cuda, output_cuda)

    # Copy the result back to the host
    output = output_cuda.copy_to_host()

    return output

@cuda.jit
def correlation_cuda_kernel(kernel, image, output):
    x, y = cuda.grid(2)

    tidx = cuda.threadIdx.x
    tidy = cuda.threadIdx.y

    rows = kernel.shape[0]
    cols = kernel.shape[1]

    print(rows, cols)

    sK = cuda.shared.array(shape=(rows, cols), dtype=np.float64)

    if tidx < kernel.shape[0] and tidy < kernel.shape[1]:  # first <m>X<n> threads copies the kernel to shmem as well
        sK[tidx, tidy] = kernel[tidx, tidy]

    cuda.syncthreads()

    sI = cuda.shared.array(shape=(32, 32), dtype=np.float64)

    mpad = kernel.shape[0] // 2
    npad = kernel.shape[1] // 2

    sI[tidx, tidy] = image[x + mpad, y + npad]  # Each thread copies one pixel to shmem

    cuda.syncthreads()

    if x >= output.shape[0] or y >= output.shape[1]:
        return

    for xx in range(-math.floor(kernel.shape[0] / 2), math.ceil(kernel.shape[0] / 2)):
        for yy in range(-math.floor(kernel.shape[1] / 2), math.ceil(kernel.shape[1] / 2)):
            if tidx + xx > 0 and tidx + xx < sI.shape[0] and tidy + yy > 0 and tidy + yy < sI.shape[1]:
                output[x, y] += sK[int(xx + math.floor(kernel.shape[0] / 2)), int(yy + math.floor(kernel.shape[1] / 2))] * sI[
                    int(tidx + xx), int(tidy + yy)]
            else:
                output[x, y] += sK[int(xx + math.floor(kernel.shape[0] / 2)), int(yy + math.floor(kernel.shape[1] / 2))] * image[
                    int(x + xx) + mpad, int(y + yy) + npad]

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
    # your calculations

    raise NotImplementedError("To be implemented")


def load_image(): 
    fname = 'data/image.jpg'
    pic = imageio.imread(fname)
    to_gray = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114])
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
