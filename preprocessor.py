#
#   @date:  [TODO: Today's date]
#   @author: [TODO: Student Names]
#
# This file is for the solutions of the wet part of HW2 in
# "Concurrent and Distributed Programming for Data processing
# and Machine Learning" course (02360370), Winter 2024
#
import multiprocessing
from scipy import ndimage
import numpy as np

class Worker(multiprocessing.Process):

    def __init__(self, jobs, result, training_data, batch_size):
        super().__init__()

        ''' Initialize Worker and it's members.

        Parameters
        ----------
        jobs: JoinableQueue
            A jobs Queue for the worker.
        result: Queue
            A results Queue for the worker to put it's results in.
		training_data: 
			A tuple of (training images array, image lables array)
		batch_size:
			workers batch size of images (mini batch size)
        
        You should add parameters if you think you need to.
        '''
        self.jobs = jobs
        self.result = result
        self.training_data = training_data
        self.batch_size = batch_size


    @staticmethod
    def rotate(image, angle):
        '''Rotate given image to the given angle

        Parameters
        ----------
        image : numpy array
            An array of size 784 of pixels
        angle : int
            The angle to rotate the image
            
        Return
        ------
        An numpy array of same shape
        '''

        temp_image = image.reshape(28, 28)
        rotated_img = ndimage.rotate(temp_image, angle, reshape=False)

        return rotated_img.reshape(784, 1)

    @staticmethod
    def shift(image, dx, dy):
        '''Shift given image

        Parameters
        ----------
        image : numpy array
            An array of shape 784 of pixels
        dx : int
            The number of pixels to move in the x-axis
        dy : int
            The number of pixels to move in the y-axis
            
        Return
        ------
        An numpy array of same shape
        '''

        temp_image = image.reshape(28, 28)
        shifted_image = ndimage.shift(temp_image, (dx, dy), mode='nearest', cval=0.0)

        return shifted_image.reshape(784, 1)

    @staticmethod
    def add_noise(image, noise):
        '''Add noise to the image
        for each pixel a value is selected uniformly from the 
        range [-noise, noise] and added to it. 

        Parameters
        ----------
        image : numpy array
            An array of shape 784 of pixels
        noise : float
            The maximum amount of noise that can be added to a pixel

        Return
        ------
        An numpy array of same shape
        '''

        temp_image = image.reshape(28, 28)
        random_noise = np.random.uniform(-noise, noise, size=temp_image.shape)
        noisy_image = temp_image + random_noise
        noisy_image = np.clip(noisy_image, 0, 255)

        return noisy_image.flatten()

    @staticmethod
    def skew(image, tilt):
        '''Skew the image
        By doing : result[i][j] = image[i][j + i*tilt]
        values out of range are treated as 0
        Parameters
        ----------
        image : numpy array
            An array of size 784 of pixels
        tilt : float
            The skew paramater

        Return
        ------
        An numpy array of same shape
        '''
        image = image.reshape(28, 28)
        skewed_image = np.zeros_like(image)

        for i in range(image.shape[0]):
            shifted_column = np.roll(image[:, i], int(i * tilt))
            skewed_image[:, i] = shifted_column

        return skewed_image.flatten()

    def process_image(self, image):
        '''Apply the image process functions
		Experiment with the random bounds for the functions to see which produces good accuracies.

        Parameters
        ----------
        image: numpy array
            An array of size 784 of pixels

        Return
        ------
        An numpy array of same shape
        '''
        angle = np.random.uniform(-10, 10)
        image = self.rotate(image, int(angle))

        # Apply shifting
        dx = np.random.randint(-5, 5)
        dy = np.random.randint(-5, 5)
        image = self.shift(image, dx, dy)

        # Apply adding noise
        noise = np.random.uniform(0, 10)
        image = self.add_noise(image, noise)

        # Apply skewing
        tilt = np.random.uniform(-0.05, 0.05)
        image = self.skew(image, tilt)

        return image

    def run(self):
        '''Process images from the jobs queue and add the result to the result queue.
		Hint: you can either generate (i.e sample randomly from the training data)
		the image batches here OR in ip_network.create_batches
        '''

        while True:
            next_job = self.jobs.get()
            if next_job is None:
                self.jobs.task_done()
                break
            batch_id = next_job['batch ID']
            images_id = next_job['indices ID']
            batch_images = np.empty((self.batch_size, 784), dtype=np.float32)
            batch_labels = np.empty((self.batch_size, 10), dtype=np.float64)
            for i, image_id in enumerate(images_id):
                next_image = self.training_data[0][image_id]
                augmented_image = self.process_image(next_image)
                next_label = self.training_data[1][image_id]
                batch_images[i] = augmented_image
                batch_labels[i] = next_label
            self.jobs.task_done()
            self.result.put((batch_images, batch_labels))