#
#   @date:  [TODO: Today's date]
#   @author: [TODO: Student Names]
#
# This file is for the solutions of the wet part of HW2 in
# "Concurrent and Distributed Programming for Data processing
# and Machine Learning" course (02360370), Winter 2024
#
from multiprocessing import Pipe, Lock
class MyQueue(object):

    def __init__(self):
        ''' Initialize MyQueue and it's members.
        '''
        self.receiver, self.sender = Pipe()
        self.lock = Lock()

    def put(self, msg):
        '''Put the given message in queue.

        Parameters
        ----------
        msg : object
            the message to put.
        '''
        self.sender.send(msg)

    def get(self):
        '''Get the next message from queue (FIFO)
            
        Return
        ------
        An object
        '''

        self.lock.acquire()
        msg = self.receiver.recv()
        self.lock.release()

        return msg

    def empty(self):
        '''Get whether the queue is currently empty
            
        Return
        ------
        A boolean value
        '''

        return not self.receiver.poll()