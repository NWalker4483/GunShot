import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
def get_spectrogram(data, plot = False):
    """
    Function to compute a spectrogram.
    
    Argument:
    predictions -- one channel / dual channel audio data as numpy array

    Returns:
    pxx -- spectrogram, 2-D array, columns are the periodograms of successive segments.
    """
    nfft = 200 # Length of each window segment
    fs = 8000 # Sampling frequencies
    noverlap = 120 # Overlap between windows
    nchannels = data.ndim
    if nchannels != 1: 
        data = data[:,0]
    if not plot:
        pxx, _, _ = mlab.specgram(data, nfft, fs, noverlap = noverlap)
    else:
        pxx, _, _, _ = plt.specgram(data, nfft, fs, noverlap = noverlap)
    return pxx
from keras.utils import np_utils

def extract_chunks(data):
    clips = [] 
    i = 0 
    limit = len(data)
    while i < len(data):
        while data[i] == 0:
            i+=1
            if i == limit:
                break 
        b = i
        if i == limit:
                break 
        while data[i] == 1:
            i+=1
            if i == limit:
                break 
        clips.append((b,i))
    return clips
def resize_mask(mask,size = 100):
    new_mask = np.zeros(size)
    for start, end in extract_chunks(mask):
        new_start, new_end = (start / len(mask)) * size, (end / len(mask)) * size
        new_mask[int(new_start):int(new_end)] = 1
    return new_mask
from tqdm import tqdm


def LoadData(source_dir = "DataLabels",Ty = 479):
    print("Loading....\n")
    X, Y = [], []
    for fn in tqdm(os.listdir(source_dir)):
        _Y = np.load(source_dir+"/"+fn)
        #_Y = resize_mask(_Y,Ty)
        _Y = _Y.reshape(len(_Y),1)
        Y.append(_Y)
        try:
            sound, _ = librosa.load("FusedSounds/"+ fn[:-11] + ".wav")
        except:
            sound, _ = librosa.load("RawSounds/"+ fn[:-11] + ".wav")
        _X = get_spectrogram(sound)
        _X = _X.swapaxes(0,1)#_X = np.expand_dims(_X, axis=0)
        X.append(_X)
        # break
        return np.array(X), np.array(Y) 
if __name__ == "__main__":
    X, Y = LoadData()


