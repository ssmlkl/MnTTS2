

import csv
import os
import wave

import numpy as np

file_dir = './spk_aodenggaowa/'  #你的文件路径
files = os.listdir(file_dir)

datas = []
for wave_name in files:
    count = 0
    if(wave_name[-1:] == "t"):
        with open(file_dir + wave_name, encoding='utf-8') as f:
            for line in f:
                print(wave_name[:-4]+"|"+line)
                datas.append(wave_name[:-4]+"|"+line)

f.close()

with open("./aodenggaowa.csv","a+", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE, quotechar=None, escapechar="")
        for data in datas:
            writer.writerow([data])
csvfile.close()