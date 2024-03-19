#
# This file is NOT for submission!
#
import numpy as np
import matplotlib.pyplot as plt
import pickle
import timeit
from filters import *
from scipy.signal import convolve2d
import imageio
import os

# 7X7
edge_kernel = np.array([[-3/18, -2/13, -1/10, 0, 1/10, 2/13, 3/18],
                        [-3/13, -2/8, -1/5, 0, 1/5, 2/8, 3/13],
                        [-3/10, -2/5, -1/2, 0, 1/2, 2/5, 3/10],
                        [-3/9, -2/4, -1/1, 0, 1/1, 2/4, 3/9],
                        [-3/10, -2/5, -1/2, 0, 1/2, 2/5, 3/10],
                        [-3/13, -2/8, -1/5, 0, 1/5, 2/8, 3/13],
                        [-3/18, -2/13, -1/10, 0, 1/10, 2/13, 3/18]])


flipped_edge_kernel = np.array([[3/18, 2/13, 1/10, 0, -1/10, -2/13, -3/18],
                        [3/13, 2/8, 1/5, 0, -1/5, -2/8, -3/13],
                        [3/10, 2/5, 1/2, 0, -1/2, -2/5, -3/10],
                        [3/9, 2/4, 1/1, 0, -1/1, -2/4, -3/9],
                        [3/10, 2/5, 1/2, 0, -1/2, -2/5, -3/10],
                        [3/13, 2/8, 1/5, 0, -1/5, -2/8, -3/13],
                        [3/18, 2/13, 1/10, 0, -1/10, -2/13, -3/18]])

# 5X5
blur_kernel = np.array([[1/25, 1/25, 1/25, 1/25, 1/25],
                        [1/25, 1/25, 1/25, 1/25, 1/25],
                        [1/25, 1/25, 1/52, 1/25, 1/25],
                        [1/25, 1/25, 1/25, 1/25, 1/25],
                        [1/25, 1/25, 1/25, 1/25, 1/25]])

# 3X3
shapen_kernel = np.array([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]])


def get_image(): 
    fname = 'data/lena.dat'
    f = open(fname, 'rb')
    lena = np.array(pickle.load(f))
    f.close()
    return np.array(lena[175:390, 175:390])


# Note: Use this on your local computer to better understand what the correlation does.
def show_image(image):
    """ Plot an image with matplotlib

    Parameters
    ----------
    image: list
        2d list of pixels
    """
    plt.imshow(image, cmap='gray')
    plt.show()


def corr_comparison():
    """ Compare correlation functions run time.
    """
    image = get_image()

    def timer(kernel, f):
        return min(timeit.Timer(lambda: f(kernel, image)).repeat(10, 1))

    def correlation_cpu(kernel, image):
        return convolve2d(image, kernel, mode='same')
    
    # cpu_time_3x3 = timer(shapen_kernel, correlation_cpu)
    # numba_time_3x3 = timer(shapen_kernel, correlation_numba)
    # gpu_time_3x3 = timer(shapen_kernel, correlation_gpu)
    # print('CPU 3X3 kernel:', cpu_time_3x3)
    # print('Numba 3X3 kernel:', numba_time_3x3)
    # print('CUDA 3X3 kernel:', gpu_time_3x3)
    # print("---------------------------------------------")
    #
    # cpu_time_5x5 = timer(blur_kernel, correlation_cpu)
    # numba_time_5x5 = timer(blur_kernel, correlation_numba)
    # gpu_time_5x5 = timer(blur_kernel, correlation_gpu)
    # print('CPU 5X5 kernel:', cpu_time_5x5)
    # print('Numba 5X5 kernel:', numba_time_5x5)
    # print('CUDA 5X5 kernel:', gpu_time_5x5)
    # print("---------------------------------------------")
    #
    # cpu_time_7x7 = timer(flipped_edge_kernel, correlation_cpu)
    # numba_time_7x7 = timer(edge_kernel, correlation_numba)
    # gpu_time_7x7 = timer(edge_kernel, correlation_gpu)
    # print('CPU 7X7 kernel:', cpu_time_7x7)
    # print('Numba 7X7 kernel:', numba_time_7x7)
    # print('CUDA 7X7 kernel:', gpu_time_7x7)
    #
    # print("---------------------------------------------")
    # print('scipy-Numba 3x3 kernel speedup: ', cpu_time_3x3 / numba_time_3x3)
    # print('scipy-GPU 3x3 kernel speedup: ', cpu_time_3x3 / gpu_time_3x3)
    # print("---------------------------------------------")
    #
    # print('scipy-Numba 5x5 kernel speedup: ', cpu_time_5x5 / numba_time_5x5)
    # print('scipy-GPU 5x5 kernel speedup: ', cpu_time_5x5 / gpu_time_5x5)
    # print("---------------------------------------------")
    # print('scipy-Numba 7x7 kernel speedup: ', cpu_time_7x7 / numba_time_7x7)
    # print('scipy-GPU 7x7 kernel speedup: ', cpu_time_7x7 / gpu_time_7x7)
    # print("---------------------------------------------")



if __name__ == '__main__':
    os.environ['NUMBAPRO_NVVM'] = '/usr/local/cuda-9.0/nvvm/lib64/libnvvm.so'
    os.environ['NUMBAPRO_LIBDEVICE'] = '/usr/local/cuda-9.0/nvvm/libdevice/'
    corr_comparison()

    res_ker1 = sobel_kernel_first()
    show_image(res_ker1)

    res_ker2 = sobel_kernel_second()
    show_image(res_ker2)

    res_ker3 = sobel_kernel_third()
    show_image(res_ker3)

    # res_cpu = sobel_operator_cpu_conv()
    # show_image(res_cpu)
