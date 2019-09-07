import os 
import librosa
import matplotlib.pyplot as plt 
import numpy as np 
from scipy.ndimage.morphology import binary_dilation
source_dir = "RawSounds"
#
def overlay(background,active, position = 0):
    background[position:position+len(active)] += active
    return background
# Extracting gunshot clips to do hotword training 
def extract_clips(data):
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
for i in os.listdir(source_dir):
    if "gun_shot" in i:
        print(i)
        raw_sound, sample_rate  = librosa.load(source_dir+"/"+i)
        raw_sound = np.array(raw_sound)
        labels = np.zeros_like(raw_sound)

        silence_threshold = max(raw_sound)/2.3
        shots = raw_sound.copy()
        mask = np.ones_like(raw_sound)
        mask[abs(raw_sound) <= silence_threshold] = 0
        mask = binary_dilation(mask, iterations=3000)
        if sum(mask) == len(raw_sound):
            continue
        shot = 0 
        base_name = i.split(".")[1]
        for start,stop in extract_clips(mask):
            librosa.output.write_wav("GunClips/gun_shot."+base_name+".clip."+str(shot)+".wav",raw_sound[start:stop],sample_rate)
            shot += 1
        np.save("DataLabels/gun_shot."+base_name+".labels.npy", mask)   
        """
        ##For Viz###
        shots[mask == 0] = 0 
        plt.plot(range(len(raw_sound)),raw_sound,"b",range(len(raw_sound)),shots,"g")
        plt.plot(range(len(raw_sound)),[silence_threshold,]*len(raw_sound),"r--")
        plt.show()
        ############
        """
#### Fuse GunShots in BackGround Noise && Generate Label Files
source_files = os.listdir("GunClips")
sample_rate = 22050 
num = 0 
for i in os.listdir(source_dir):
    if "gun_shot" not in i:
        print(i)
        num += 1
        background,_ = librosa.load(source_dir+"/"+i)
        labels = np.zeros_like(background)
        #Generate points to insert gunshots
        buffer = -1
        fails = -1 
        while buffer <= 0:
            fails += 1 
            # Select up to 10 random gunshot files
            curr_source = [source_files[i] for i in np.random.choice(len(source_files),np.random.randint(0,10))]
            # Load these files into memory
            audio_samples = [librosa.load("GunClips/"+j)[0] for j in curr_source]
            # If the files were stacked end to end in the background noise how much space is left remaining
            buffer = len(background) - sum([len(sample) for sample in audio_samples])
            if fails == 5: 
                break 
        if fails == 5:
            print("Fails")
            continue
        start = 0
        for sample in audio_samples:
            end = np.random.randint(start,start+buffer)
            background = overlay(background, sample ,position = end)
            labels[end:end+len(sample)] = 1
            buffer -= (end - start)  
            start = end 
        librosa.output.write_wav("FusedSounds/"+i.split(".")[0]+"."+str(num)+".fused.wav",background,sample_rate)
        np.save("DataLabels/"+i.split(".")[0]+"."+str(num)+".fused.labels.npy", labels)   