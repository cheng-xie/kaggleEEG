import os
import sys
import numpy as np

# Traverse through whole directory and compile all as one numpy array
class EEGDataLoader:
    def __init__(self, trainf, testf, batch_size, window_size, stride):
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
        # self.indices = np.arrange(0,)


    def load(self, trainf, testf):
        self.train_array = []
        self.test_array = []
        if(os.path.isdir(trainf)):
            for filename in os.listdir(trainf):
                base, ext = os.path.splitext(filename) 
                if( filename.endswith('.npy') ):
                    _, _, y = re.findall('(\d+)\_(\d+)\_(\d+)', base)[0]
                    if (y == '0'):
                        try:
                            self.train_array0 = np.append(self.train_array0, [np.load(filename)['data']], axis = 0)
                            print( 'Loaded:' + base )
                        except:
                            print( 'Could not load:' + base )
                            pass
                    elif (y == '1'):
                        try:
                            self.train_array1 = np.append(self.train_array1, [np.load(filename)['data']], axis = 0)
                            print( 'Loaded:' + base )
                        except:
                            print( 'Could not load:' + base )
                            pass                       
        '''
        if(os.path.isdir(testf)):
                for filename in os.listdir(testf):
                base, ext = os.path.splitext() 
                if( filename.endswith('.npy') ):
                try:
                sequence = np.load(filename)
                self.test_array = np.append(self.test_array, [sequence], axis = 0)
                except:
                print( 'Could not load:' + base )
                pass
        '''

    def next(self):
        # generate random batch, make sure to keep proportion between 
        # positive and negative samples
        n = self.batch_size/2
        batch_x = np.zeros((self.batch_size, self.window_size, 16))
        indices_0 = np.random.randint(len(self.train_array0), size = n)
        indices_1 = np.random.randint(len(self.train_array1), size = n)
        
        for i in xrange(n):
            t_steps = len(self.train_array0[indices_0[i]])#.shape[0]
            index = np.random.randint(-1, (t_steps-1-window_size)/stride)
            if (index == -1):
                batch_x[i] = self.train_array0[indices_0[i]][index*stride:index*stride+self.window_size]
            else:
                batch_x[i] = self.train_array0[indices_0[i]][t_steps-self.window_size:t_steps]
            
        for i in xrange(n):
            t_steps = len(self.train_array1[indices_1[i]])#.shape[0]
            index = np.random.randint(-1, (t_steps-1-window_size)/stride)
            if (index == -1):
                batch_x[n+i+1] = self.train_array1[indices_1[i]][index*stride:index*stride+self.window_size]
            else:
                batch_x[n+i+1] = self.train_array1[indices_1[i]][t_steps-self.window_size:t_steps]
        
        batch_y = np.zeros(self.batch_size, 2)
        batch_y[0:n, 0] = 1
        batch_y[n:2*n, 1] = 1

        return batch_x, batch_y
