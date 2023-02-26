import os
import numpy as np

file_dir = './durations'  #你的文件路径
files = os.listdir(file_dir)

for dur_name in files:

    current_dur = np.load("./durations/"+dur_name, allow_pickle=True, encoding="latin1")
    num_dur = current_dur.shape[0]  # 时间的长度

    current_name = dur_name[:-14]
    # print(current_name)

    current_ids = np.load("./ids/"+current_name+"-ids.npy", allow_pickle=True, encoding="latin1")
    num_ids = current_ids.shape[0]  # 时间的长度

    if(num_dur != num_ids):
        print(dur_name)

    if(num_dur > num_ids):
        print(str(num_dur)+"    "+str(num_ids))
        count = num_dur - num_ids
        current_dur = current_dur[:-count]
        np.save("./durations/"+dur_name, current_dur)


    if(num_dur < num_ids):
        print(str(num_dur) + "    " + str(num_ids))
        count = num_ids - num_dur
        current_ids = current_ids[:-count]
        np.save("./ids/"+current_name+"-ids.npy", current_ids)