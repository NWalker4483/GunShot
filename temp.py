sound_names = ["air conditioner","car horn","children playing","dog bark","drilling","engine idling",
               "gun shot","jackhammer","siren","street music"]
import os 
sound_names = ["_".join(sound_name.split()) for sound_name in sound_names]
for i in sound_names:
    num = 0
    for j in os.listdir("."):
        if i in j:
            num += 1
            try:
                os.rename(j,i+"."+str(num)+".wav")
            except:
                pass