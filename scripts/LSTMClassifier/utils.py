import os
import sys
import numpy as np
import re
import math
import random

'''
The strategy of the data loader is to load all the 
'''

# Traverse through whole directory and compile all as one numpy array
class EEGDataLoader:
    
    MAX_LOAD_FILES = 200

    def __init__(self, traindir, testdir, mini_batch_size, window_size, stride):
        # Save paramaters
        self.mini_batch_size = mini_batch_size
        self.window_size = window_size
        self.stride = stride
        
        # scan files and add to filenames
        self.filenames = {0:[], 1:[]}
        if(os.path.isdir(trainf)):
            for filename in os.listdir(trainf):
                base, ext = os.path.splitext(os.path.basename(filename)) 
                if( filename.endswith('.npy') ):
                    _, _, y = re.findall('(\d+)\_(\d+)\_(\d+)', base)[0]
                    if (y == '0'):
                        #try:
                        self.filenames[0].append(os.path.join(trainf, filename))
                        print( 'Loaded:' + base + ' ' + str(self.train_array0[-1].shape))

                    elif (y == '1'):
                        #try:
                        #self.train_array1.append(np.load(filename)['data'])
                        self.filenames[1].append(os.path.join(trainf, filename))
                        print( 'Loaded:' + base + ' ' + str(self.train_array1[-1].shape))

        # partition filenames into batches for first epoch
        self.batched_filenames = [] 
        self.next_epoch()

        # setup data lists for first batch 
        self.train_array = ([],[])
        self.test_array = []
        self.cur_batch = 0
        # store indices for all data slices so we can shuffle
        self.batch_indices = ([],[])
        self.load_batch()

        #
        self.mini_batch_index = 0

    def load_batch(self):
        '''
        Loads the files of the current batch cycle.
        Sets up the indices for shuffling minibatches.
        '''
        self.train_array = ([],[])
        self.batch_indices = ([],[])

        # iterate over files, add them 
        for i, filename in enumerate(self.batched_filenames[self.cur_batch][0]):
            self.train_array[0].append(np.load(filename)['data'][()])
            n = self.train_array[0][-1].shape[0]
            windows = (n-1-self.window_size)/self.stride
            self.batch_indices[0] += [ (i,x) for x in xrange(-1, windows+1) ]
        
        for i, filename in enumerate(self.batched_filenames[self.cur_batch][1]):
            self.train_array[1].append(np.load(filename)['data'][()])
            n = self.train_array[1][-1].shape[0]
            windows = (n-1-self.window_size)/self.stride
            self.batch_indices[1] += [ (i,x) for x in xrange(-1, windows+1) ]
        
        random.shuffle(self.batch_indices[0])
        random.shuffle(self.batch_indices[1])

    def next_epoch(self):
        '''
        Called after an epoch has been completed. 
        Resets data indices and repartitions filenames for batches.
        '''
        # TODO: Zero out indices and clean up stuff


        # scramble filenames and repartition
        random.shuffle(self.filenames[0])
        random.shuffle(self.filenames[1])

        # compute partition for the files to be loaded 
        n_pos = len(self.filenames[1])
        n_neg = len(self.filenames[0])        
        class_ratio = float(n_neg) / n_pos
        num_batches = (n_pos + n_neg) / MAX_LOAD_FILES
        batched_0s = list(chunks(self.filenames[0]))
        batched_1s = list(chunks(self.filenames[1]))
        self.batched_filenames = zip(batched_0s, batched_1s)

    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
            for i in range(0, len(l), n):
                yield l[i:i + n]

    def load_test_folder(self, testf):
        self.test_array = []
        if(os.path.isdir(trainf)):
            for filename in os.listdir(trainf):
                base, ext = os.path.splitext(os.path.basename(filename)) 
                if( filename.endswith('.npy') ):
                    _, _, y = re.findall('(\d+)\_(\d+)\_(\d+)', base)[0]
                    if (y == '0'):
                        #try:
                        self.train_array0.append(np.load(os.path.join(trainf, filename))['data'][()])
                        print( 'Loaded:' + base + ' ' + str(self.train_array0[-1].shape))
   
                   elif (y == '1'):
                        #try:
                        #self.train_array1.append(np.load(filename)['data'])
                        self.train_array1.append(np.load(os.path.join(trainf, filename))['data'][()])
                        print( 'Loaded:' + base + ' ' + str(self.train_array1[-1].shape))
        
    def next_mini_batch(self):
        # generate random batch, make sure to keep proportion between 
        # positive and negative samples
        n = self.mini_batch_size/2
        batch_x = np.zeros((self.mini_batch_size, self.window_size, 16))
       
         
        for i in xrange(n):
            t_steps = self.train_array0[indices_0[i]].shape[0]
            index =
            if (index == -1):
                batch_x[i] = self.train_array0[indices_0[i]][t_steps-self.window_size:t_steps]
            else:
                batch_x[i] = self.train_array0[indices_0[i]][index*self.stride:index*self.stride+self.window_size]
        
        # To maintain fifty fifty ratio, this will hit the end sooner
        # We need to oversample
        for i in xrange(n):
            t_steps = self.train_array1[indices_1[i]].shape[0]
            index = 
            if (index == -1):
                batch_x[n+i] = self.train_array1[indices_1[i]][t_steps-self.window_size:t_steps]
            else:
                batch_x[n+i] = self.train_array1[indices_1[i]][index*self.stride:index*self.stride+self.window_size]
        
        batch_y = np.zeros((self.batch_size, 2), dtype=float)
        batch_y[0:n, 0] = 1
        batch_y[n:2*n, 1] = 1

        # Trust me will shuffle the same way
        rng_state = np.random.get_state()
        np.random.shuffle(batch_x)
        np.random.set_state(rng_state)
        np.random.shuffle(batch_y)
        
        return batch_x, batch_y

    def test_batch(self):
        # return
