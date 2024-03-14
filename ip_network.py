#
#   @date:  [TODO: Today's date]
#   @author: [TODO: Student Names]
#
# This file is for the solutions of the wet part of HW2 in
# "Concurrent and Distributed Programming for Data processing
# and Machine Learning" course (02360370), Winter 2024
#
import matplotlib.pyplot as plt
import math
import os
import sys

from network import *
from preprocessor import *
from multiprocessing import JoinableQueue, Queue


def plot_minibatch(data, labels, indices, batch_size, k):
    def one_hot_to_number(x):
        return np.where(x == 1)[0].item()

    rows = math.ceil(math.sqrt(batch_size))
    cols = math.ceil(math.sqrt(batch_size))
    fig, axes = plt.subplots(rows, cols)
    plt.tight_layout()
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[indices[i]].reshape((28, 28)))
        ax.set_title(one_hot_to_number(labels[indices[i]]))
    plt.show()
    fig.savefig(f'original_minibatch{k}')

class IPNeuralNetwork(NeuralNetwork):

    def fit(self, training_data, validation_data=None):
        '''
        Override this function to create and destroy workers
        '''
        # 1. Create Workers
        # (Call Worker() with self.mini_batch_size as the batch_size)
        self.jobs = JoinableQueue()
        self.result = Queue()
        # 2. Set jobs
        workers = [ Worker(self.jobs, self.result, training_data, self.mini_batch_size) for i in range(int(os.environ['SLURM_CPUS_PER_TASK'])) ]

        for w in workers:
            w.start()

        # Call the parent's fit. Notice how create_batches is called inside super.fit().
        super().fit(training_data, validation_data)

        # 3. Stop Workers
        for _ in range(int(os.environ['SLURM_CPUS_PER_TASK'])):
            self.jobs.put(None)

        self.jobs.join()

        [w.join() for w in workers]

        def create_batches(self, data, labels, batch_size):
            '''
            Override this function to return self.number_of_batches batches created by workers
            Hint: you can either generate (i.e sample randomly from the training data) the image batches here OR in Worker.run()
            '''
            for k in range(self.number_of_batches):
                indices = random.sample(range(0, data.shape[0]), batch_size)

                job = {'batch ID': k, 'indices ID': indices}
                self.jobs.put(job)

            batches = []
            for k in range(self.number_of_batches):
                batches.append(self.result.get())

            return batches
