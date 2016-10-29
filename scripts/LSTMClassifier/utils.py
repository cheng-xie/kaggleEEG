import os
import sys
import numpy as np
import re

# Traverse through whole directory and compile all as one numpy array
class EEGDataLoader:

    def __init__(self, traindir, testdir, batch_size, window_size, stride):
        # Save paramaters
        self.batch_size = batch_size
        self.window_size = window_size
        self.stride = stride
        
        # setup data ndarrays
        self.train_array0 = []
        self.train_array1 = []
        self.test_array = []
        self.load(trainf, testf)

        # TODO: figure out the batching
        
        self.indices = np.arrange(0,)


    def load(self, traindir, testf):
       

    def load_train_folder(self, trainf):
        self.train_array0 = []
        self.train_array1 = []
        self.test_array0 = []
        self.test_array1 = []
        if(os.path.isdir(trainf)):
            for filename in os.listdir(trainf):
                base, ext = os.path.splitext(os.path.basename(filename)) 
                if( filename.endswith('.npy') ):
                    _, _, y = re.findall('(\d+)\_(\d+)\_(\d+)', base)[0]
                    if (y == '0'):
                        #try:
                        self.train_array0.append(np.load(os.path.join(trainf, filename))['data'][()])
                        print( 'Loaded:' + base + ' ' + str(self.train_array0[-1].shape))
                        #except:
                        #    print( 'Could not load:' + base )
                        #    pass
                    elif (y == '1'):
                        #try:
                        #self.train_array1.append(np.load(filename)['data'])
                        self.train_array1.append(np.load(os.path.join(trainf, filename))['data'][()])
                        print( 'Loaded:' + base + ' ' + str(self.train_array1[-1].shape))
        
                        #except:
                        #    print( 'Could not load:' + base )
                        #    pass
        
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
                        #except:
                        #    print( 'Could not load:' + base )
                        #    pass
                    elif (y == '1'):
                        #try:
                        #self.train_array1.append(np.load(filename)['data'])
                        self.train_array1.append(np.load(os.path.join(trainf, filename))['data'][()])
                        print( 'Loaded:' + base + ' ' + str(self.train_array1[-1].shape))
        
                        #except:
                        #    print( 'Could not load:' + base )
                        #    pass

    def next_batch(self):
        # generate random batch, make sure to keep proportion between 
        # positive and negative samples
        n = self.batch_size/2
        batch_x = np.zeros((self.batch_size, self.window_size, 16))
        indices_0 = np.random.randint(len(self.train_array0), size = n)
        indices_1 = np.random.randint(len(self.train_array1), size = n)
        
        for i in xrange(n):
            t_steps = self.train_array0[indices_0[i]].shape[0]
            index = np.random.randint(-1, (t_steps-1-self.window_size)/self.stride)
            if (index == -1):
                batch_x[i] = self.train_array0[indices_0[i]][t_steps-self.window_size:t_steps]
            else:
                batch_x[i] = self.train_array0[indices_0[i]][index*self.stride:index*self.stride+self.window_size]
            
        for i in xrange(n):
            t_steps = self.train_array1[indices_1[i]].shape[0]
            index = np.random.randint(-1, (t_steps-1-self.window_size)/self.stride)
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
