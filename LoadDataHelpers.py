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

def LoadData(source_dir = "DataLabels"):
    print("Loading....\n")
    X, Y = [], []
    for fn in os.listdir(source_dir):
        Y.append(np.load(source_dir+"/"+fn))
        try:
            sound, _ = librosa.load("FusedSounds/"+ fn[:-11] + ".wav")
        except:
            sound, _ = librosa.load("RawSounds/"+ fn[:-11] + ".wav")
        X.append(get_spectrogram(sound))
    return X, Y 

