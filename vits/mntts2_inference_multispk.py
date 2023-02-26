import os
import re
import json
import math
import torch
import soundfile as sf
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from scipy.io.wavfile import write


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

hps = utils.get_hparams_from_file("./configs/mntts2.json")
net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model).cuda()
_ = net_g.eval()

print("n_speakers:",hps.data.n_speakers)

_ = utils.load_checkpoint("/home/LiuruiGrp/2022lkl/project/vits16bit/logs/mntts2/G_388000.pth", net_g, None)

"""mntts2/01/01_1_006797"""
Synthesiz_txt="/home/LiuruiGrp/2022lkl/project/vits16bit/inf.txt"
with open(Synthesiz_txt,encoding="utf-8") as f:
    for line in f:
        parts=line.strip().split('|')
        path=parts[0]
        file=path.strip().split('/')[2]
        filename=os.path.splitext(file)[0]
        print("filename:",filename)
        spk_id=parts[1]
        print("spk_id:",spk_id)
        text=parts[2]
        print("text:",text)
        with torch.no_grad():
            x_tst = get_text(text,hps).cuda().unsqueeze(0)
            x_tst_lengths = torch.LongTensor([get_text(text,hps).size(0)]).cuda()
            spk = torch.LongTensor([int(spk_id)]).cuda()
            audio = net_g.infer(x_tst, x_tst_lengths, sid=spk, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
            save_path=os.path.join("/home/LiuruiGrp/2022lkl/project/vits16bit/tts_wav",spk_id,filename+".wav")
            print("savepath",save_path)
            sf.write(save_path,audio,hps.data.sampling_rate)

print("----------------Finished----------------")
