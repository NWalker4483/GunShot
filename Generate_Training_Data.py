import os  
import librosa
import matplotlib.pyplot as plt 
import numpy as np 
from scipy.ndimage.morphology import binary_dilation
from tqdm import tqdm
from SoundHelpers import *

# Extracting gunshot clips to do hotword training 
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
def divide_chunks(l, n): 
    # looping till length l 
    data = []
    for i in range(0, len(l), n):  
        data.append(l[i:i + n])
    return data 
def ExtractGunClips(source_dir = "RawSounds"):
    print("Extracting Clips...")
    for i in tqdm(os.listdir(source_dir)):
        if "gun_shot" in i:
            raw_sound, sample_rate  = librosa.load(source_dir+"/"+i)
            raw_sound = np.array(raw_sound)

            silence_threshold = max(raw_sound)/2.3
            mask = np.ones_like(raw_sound)
            mask[abs(raw_sound) <= silence_threshold] = 0
            mask = binary_dilation(mask, iterations=3000)

            if sum(mask) == len(raw_sound):
                continue
            shot = 0 
            base_name = i.split(".")[1]
            for start,stop in extract_chunks(mask):
                librosa.output.write_wav("GunClips/gun_shot."+base_name+".clip."+str(shot)+".wav",raw_sound[start:stop],sample_rate)
                shot += 1
            np.save("DataLabels/gun_shot."+base_name+".labels.npy", mask)   
            
            ##For Viz###
            shots = raw_sound.copy()
            shots[mask == 0] = 0 
            plt.plot(range(len(raw_sound)),raw_sound,"b",range(len(raw_sound)),shots,"g")
            plt.plot(range(len(raw_sound)),[silence_threshold,]*len(raw_sound),"r--")
            plt.show()
            ############
def FuseGunShots(gun_source = "GunClips"):# in BackGround Noise && Generate Label Files
    source_files = os.listdir(gun_source)
    sample_rate = 22050 
    clip_length = 7 #Seconds 
    print("Fusing Shots...")
    for i in tqdm(os.listdir(source_dir)):
        if "gun_shot" not in i:
            source,_ = librosa.load(source_dir+"/"+i)
            clips = divide_chunks(source,(sample_rate * clip_length))
            if len(clips[-1]) < (sample_rate * clip_length)/2: 
                # The leftover sound is less than half the 
                # target length then discard it 
                clips = clips[:-1]
            else: # Append With Silence
                clips[-1] = np.append(clips[-1], ((sample_rate * clip_length) - len(clips[-1])) * [0])
            num2 = 0
            for background in clips:
                num2 += 1 
                labels = np.zeros_like(background)
                #Generate points to insert gunshots
                buffer = -1
                fails = -1 
                while buffer <= 0 and fails <= 5:
                    fails += 1 
                    # Select up to 10 random gunshot files
                    curr_source = [source_files[i] for i in np.random.choice(len(source_files),np.random.randint(0,5))]
                    # Load these files into memory
                    audio_samples = [librosa.load(gun_source+"/"+j)[0] for j in curr_source]
                    # If the files were stacked end to end in the background noise how much space is left remaining
                    buffer = len(background) - sum([len(sample) for sample in audio_samples])
                start = 0
                for sample in audio_samples:
                    end = np.random.randint(start,start+buffer)
                    background = overlay(background, sample ,position = end)
                    labels[end:end+len(sample)] = 1
                    buffer -= (end - start)  
                    start = end 
                librosa.output.write_wav("FusedSounds/"+".".join(i.split(".")[:2])+"."+str(num2)+".fused.wav",background,sample_rate)
                np.save("DataLabels/"+".".join(i.split(".")[:2])+"."+str(num2)+".fused.labels.npy", labels)   
if __name__ == "__main__":
    source_dir = "RawSounds"
    try:
        os.mkdir("FusedSounds")
        os.mkdir("DataLabels")
        os.mkdir("GunClips")
    except:
        pass
    ExtractGunClips()
    #FuseGunShots()
    pass