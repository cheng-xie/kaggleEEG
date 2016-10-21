import os
import sys
import numpy as np

# Traverse through whole directory and compile all as one numpy array
class EEGDataLoader:
    def __init__(trainf, testf, batch_size, window_size, stride):
        # Save paramaters
        self.batch_size = batch_size
        self.window_size = window_size
        self.stride = stride
        
        # setup data ndarrays
        self.train_array = []
        self.test_array = []
        self.load(trainf, testf)

        # figure out the batching
        # setup indexed sliding window list for each file
        self.indices = []


    def load(trainf, testf):
        self.train_array = []
        self.test_array = []
        if(os.path.isdir(trainf)):
            for filename in os.listdir(trainf):
                base, ext = os.path.splitext() 
                if( filename.endswith('.npy') ):
                    try:
                        sequence = np.load(filename)
                        self.train_array = np.append(self.train_array, [sequence], axis = 0)
                    except:
                        print( 'Could not load:' + base )
                        pass

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
    
    def next()

def main(argv):
    inputf = ''
    outputdir = ''
    try:
        opts, args = getopt.getopt(argv, "i:o:", ["ifile=","odir="])
    except<F10> getopt.GetoptError:
        print 'test.py -i <inputfile/dir> -o <outputdir>'
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-i", "--ifile"):
            inputf = arg
        elif opt in ("-o", "--odir"):
            outputdir = arg
    
    if(not inputf or not outputf):
        print 'test.py -i <inputfile/dir> -o <outputdir>'
        sys.exit(2)

    if(os.path.isdir(inputf)):
        i = 0
        success = 0
        for filename in os.listdir(inputf):
            base, _ = os.path.splitext(os.path.basename(filename))
            if( filename.endswith('.mat') and (base + '.npy' not in os.listdir(outputdir)) ):
                i+=1
                print(str(i) + ' Converting: ' + filename)
                if(save_conversion( os.path.join(inputf, filename), os.path.join(outputdir, base + '.npy'))):
                    success+=1
        print('Total successful: ' + str(success) + '/' + str(i))
    else:
        base, _ = os.path.splitext(os.path.basename(inputf))
        print('Converting: ' + inputf)
        save_conversion(inputf, os.path.join(outputdir, base + '.npy'))


if __name__ == "__main__":
    main(sys.argv[1:])
