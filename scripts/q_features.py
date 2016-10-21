import numpy as np
import pyeeg 

def single_window_q_features(eeg_window, samplerate):
    '''
    Calculates all the features for the provided eeg sample data.
    Data should be an ndarray with 
    '''
    features = []
    
    # Calculate spectral entropy
    bins = [0.1, 4, 8, 12, 30, 70, 180]
    spec_entropies = pyeeg.spectral_entropy(eeg_window, bins, samplerate)


def save_slide_window_q_features(eeg_data, window_size, stride):
    '''
    Takes as input full eeg data as ndarray.
    Splits the data into sliding window batches and recursively computes features for each window.
    Returns ndarray with features for each window.
    '''
    f_windows=[]
    
    samplef = eeg_data['iEEGsamplingRate']
    N = eeg_data['data'].shape()[1]
    
    window_ranges = 
        [(x*stride, x*stride + (window_size-1)) for x in range(0, floor((N-window_size)/stride))] + [(N-1-window_size,N-1)]
    
    for start, end in window_ranges:
        window_data = eeg_data[start : end] 
        f_window = single_window_q_features(window_data, samplef)
        np.concatenate(f_windows, [f_window])

def main(argv):
    inputf = ''
    outputdir = ''
    try:
        opts, args = getopt.getopt(argv, "i:o:", ["ifile=","odir="])
    except getopt.GetoptError:
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
