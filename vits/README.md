# VITS

## Pre-requisites
0. Python >= 3.6
0. Clone this repository
0. Install python requirements. Please refer [requirements.txt](requirements.txt)
    1. You may need to install espeak first: `apt-get install espeak`
    2. For mult-speaker setting, download and extract the VCTK dataset, and downsample wav files to 22050 Hz. Then rename or create a link to the dataset folder: `ln -s /path/to/MnTTS2/wavs_16bit MnTTS2`
0. Build Monotonic Alignment Search and run preprocessing. 
```sh
# Cython-version Monotonoic Alignment Search
cd monotonic_align
python setup.py build_ext --inplace

# Preprocessing (g2p) for MnTTS2.
# python preprocess.py --text_index 2 --filelists filelists/train.txt filelists/valid.txt filelists/test.txt
```


## Training Exmaple
```sh

# MnTTS2
python train_ms.py -c configs/mntts2.json -m mntts2
```


## Inference Example
```
python mntts2_inference_multispk.py
```
