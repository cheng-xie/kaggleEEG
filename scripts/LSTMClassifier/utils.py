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
        
        self.filenames = {0:[], 1:[]}
        # scan files and add to filenames
        if(os.path.isdir(trainf)):
            for filename in os.listdir(trainf):
                base, ext = os.path.splitext(os.path.basename(filename)) 
                if( filename.endswith('.npy') ):
                    _, _, y = re.findall('(\d+)\_(\d+)\_(\d+)', base)[0]
                    if (y == '0'):
                        self.filenames[0].append(os.path.join(trainf, filename))
                        print( 'Found:' + base)

                    elif (y == '1'):
                        self.filenames[1].append(os.path.join(trainf, filename))
                        print( 'Found:' + base)

        # partition filenames into batches for first epoch
        self.batched_filenames = [] 
        self.next_epoch()

        # setup data lists for first batch 
        self.train_array = ([],[])
        self.test_array = []
        self.cur_batch = 0
        # store indices for all data slices so we can shuffle
        self.window_indices = ([],[])
        self.load_batch()

        # stores how far we are along the examples of train_array
        self.mini_batch_index = (0,0)
        

    def load_batch(self):
        '''
        Loads the files of the current batch cycle.
        Sets up the indices for shuffling minibatches.
        '''
        self.train_array = ([],[])
        self.window_indices = ([],[])

        # iterate over files, load their data and add each possible window to indices for minibatch loading 
        for i, filename in enumerate(self.batched_filenames[self.cur_batch][0]):
            self.train_array[0].append(np.load(filename)['data'][()])
            n = self.train_array[0][-1].shape[0]
            windows = (n-1-self.window_size)/self.stride
            self.window_indices[0] += [ (i,x) for x in xrange(-1, windows+1) ]
            print(str(i) + ' ' + filename + '\t' + str(self.train_array[0][-1].shape))
        
        for i, filename in enumerate(self.batched_filenames[self.cur_batch][1]):
            self.train_array[1].append(np.load(filename)['data'][()])
            n = self.train_array[1][-1].shape[0]
            windows = (n-1-self.window_size)/self.stride
            self.window_indices[1] += [ (i,x) for x in xrange(-1, windows+1) ]
            print(str(i) + ' ' + filename + '\t' + str(self.train_array[1][-1].shape))
        
        # shuffle the indices
        random.shuffle(self.window_indices[0])
        random.shuffle(self.window_indices[1])

    def next_epoch(self):
        '''
        Called after an epoch has been completed. 
        Resets data indices and repartitions filenames for batches.
        '''
        # TODO: Zero out indices and clean up stuff
        self.cur_batch = 0
        self.mini_batch_index = (0,0)
        self.train_array = ([],[])

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
        # generate random mini batch, make sure to keep proportion between 
        # will process all files for one epoch and generate all mini batches
        # will return when epoch is complete
        
        # positive and negative samples
        n = self.mini_batch_size/2
        mini_batch_x = np.zeros((self.mini_batch_size, self.window_size, 16))
        
        # check if we are done with this batch
        if ( self.mini_batch_index[0] + n >= len(self.window_indices[0]) ): 
            # if we are done check if we are done with the whole epoch
            if ( self.cur_batch + 1 >= len(self.batched_filenames) ): 
                raise StopIteration()
            self.next_batch()

        for i in xrange(n):
            seq, win_t = self.window_indices[0][self.mini_batch_index[0]]
            t_steps = self.train_array[0][seq].shape[0]
            if (index == -1):
                mini_batch_x[i] = self.train_array[0][seq][t_steps-self.window_size:t_steps]
            else:
                start = win_t * self.stride
                mini_batch_x[i] = self.train_array[0][seq][start:start+self.window_size]
            self.mini_batch_index[0] += 1
        
        # To maintain fifty fifty ratio, this will hit the end sooner
        # We need to oversample, so shuffle indices and reset iteration
        if ( self.mini_batch_index[1] + n >= len(self.window_indices[1]) ): 
            # Shuffle and reset
            random.shuffle(self.window_indices[1])
            self.mini_batch_index[1] = 0

        for i in xrange(n):
            seq, win_t = self.window_indices[1][self.mini_batch_index[1]]
            t_steps = self.train_array[1][seq].shape[0]
            if (index == -1):
                mini_batch_x[i] = self.train_array[1][seq][t_steps-self.window_size:t_steps]
            else:
                start = win_t * self.stride
                mini_batch_x[i] = self.train_array[1][seq][start:start+self.window_size]
            self.mini_batch_index[1] += 1         

        mini_batch_y = np.zeros((self.batch_size, 2), dtype=float)
        mini_batch_y[0:n, 0] = 1
        mini_batch_y[n:2*n, 1] = 1

        # Trust me will shuffle the same way
        rng_state = np.random.get_state()
        np.random.shuffle(batch_x)
        np.random.set_state(rng_state)
        np.random.shuffle(batch_y)
        
        return mini_batch_x, mini_batch_y

    def test_batch(self):
        # return
