import os
import librosa
import numpy as np

from keras.utils import np_utils
from SoundHelpers import get_spectrogram

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
    print("Loading in Data....\n")
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
        _X = _X.swapaxes(0,1)
        X.append(_X)
    return np.array(X), np.array(Y) 
if __name__ == "__main__":
    X, Y = LoadData()


