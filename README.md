# MnTTS2: An Open-Source Multi-Speaker Mongolian Text-to-Speech Synthesis Dataset
 
## Introduction
This is the experimental description of MnTTS2.

## 0) Environment Preparation

This project uses `conda` to manage all the dependencies, you should install [anaconda](https://anaconda.org/) if you have not done so. 

```bash
# Clone the repo
git clone https://github.com/ssmlkl/MnTTS2.git
cd $PROJECT_ROOT_DIR
```

### Install dependencies
```bash
conda env create -f Environment/environment.yaml
```

### Activate the installed environment
```bash
conda activate mntts2
```

## 1) Prepare MnTTS Dataset

Prepare our MnTTS2 dataset in the following format:
```
|- mntts2/
|   |- train.txt
|   |- spk_01/
|       |- file1.wav
|       |- file1.txt
|       |- ......
|       |- spk_01_train.txt
|   |- spk_02/
|       |- file1.wav
|       |- file1.txt
|       |- ......
|       |- spk_02_train.txt
|   |- spk_03/
|       |- file1.wav
|       |- file1.txt
|       |- ......
|       |- spk_03_train.txt
```

Where `spk_spkID_train.txt` has the following format: `uttID|transcription`. This is a ljspeech-like format.
And `train.txt` has the following format: `spkID|uttID|transcription`.

[The complete dataset is available from our multilingual corpus website](http://mglip.com/corpus/corpus_detail.html?corpusid=20221106113633).


## 2) Tacotron2 Preprocessing for each speaker
Take speaker 01 for example.

The preprocessing has two steps:

1. Preprocess audio features
    - Convert characters to IDs
    - Compute mel spectrograms
    - Normalize mel spectrograms to [-1, 1] range
    - Split the dataset into train and validation
    - Compute the mean and standard deviation of multiple features from the **training** split
2. Standardize mel spectrogram based on computed statistics

Please note that you will also need to check the mntts process(/home/anaconda3/envs/mntts2/lib/python3.8/site-packages/tensorflow_tts/processor/mntts.py) before you start. Since tacotron2 needs to read mntts2/spk_spkID/train.txt and fastspeech2 needs to read mntts2/train.txt. Here is an example modification:
Before starting tacotron work, you need modify mntts.py to look like the following code block:
```
    positions = {
        "wave_file": 0,
        "text": 1,
        "text_norm": 1,
    }
    train_f_name: str = "spk_01_train.txt"

    def create_items(self):
        if self.data_dir:
            with open(
                    os.path.join(self.data_dir, self.train_f_name), encoding="utf-8"
            ) as f:
                self.items = [self.split_line(self.data_dir, line, "|") for line in f]

    def split_line(self, data_dir, line, split):
        parts = line.strip().split(split)
        wave_file = parts[self.positions["wave_file"]]
        text_norm = parts[self.positions["text_norm"]]
        wav_path = os.path.join(data_dir, f"{wave_file}.wav")
        speaker_name = "spk_01"
        return text_norm, wav_path, speaker_name

```
After completing the above work, you are ready to start the tacotron2 work officially.

To reproduce the steps above:
```
CUDA_VISIBLE_DEVICES=0 tensorflow-tts-preprocess \
  --rootdir ./MnTTS2/spk_01 \
  --outdir ./tacotron2_dump/spk_01 \
  --config preprocess/mntts2_preprocess.yaml \
  --dataset mntts
```

```
CUDA_VISIBLE_DEVICES=0 tensorflow-tts-normalize \
  --rootdir ./tacotron2_dump/spk_01 \
  --outdir ./tacotron2_dump/spk_01 \
  --config preprocess/mntts2_preprocess.yaml \
  --dataset mntts
```

 



## 3) Training TacoTron2 from scratch with MnTTS dataset for each speaker

Based on the script [`train_tacotron2.py`](https://github.com/TensorSpeech/TensorFlowTTS/blob/master/examples/tacotron2/train_tacotron2.py).

 
This example code show you how to train Tactron-2 from scratch with Tensorflow 2 based on custom training loop and tf.function. 

  
Take speaker 01 for example:
First you need to refer to the instructions in 2) and modify the mntts.py file.

```bash
CUDA_VISIBLE_DEVICES=0 python examples/tacotron2/train_tacotron2.py \
  --train-dir ./tacotron2_dump/spk_01/train/ \
  --dev-dir ./tacotron2_dump/spk_01/valid/ \
  --outdir ./examples/tacotron2/exp/train.tacotron2.v1.spk_01/ \
  --config ./examples/tacotron2/conf/tacotron2.v1.yaml \
  --use-norm 1 \
  --mixed_precision 0 \
  --resume ""
```

IF you want to use MultiGPU to training you can replace `CUDA_VISIBLE_DEVICES=0` by `CUDA_VISIBLE_DEVICES=0,1,2,3` for example. You also need to tune the `batch_size` for each GPU (in config file) by yourself to maximize the performance. Note that MultiGPU now support for Training but not yet support for Decode.

In case you want to resume the training progress, please following below example command line:

```bash
--resume ./examples/tacotron2/exp/train.tacotron2.v1.spk_01/checkpoints/ckpt-100000
```

If you want to finetune a model, use `--pretrained` like this with your model filename
```bash
--pretrained pretrained.h5
```

Extract duration from alignments for FastSpeech

You may need to extract durations for student models like fastspeech. Here we use teacher forcing with window masking trick to extract durations from alignment maps:

Extract for valid set:

```
CUDA_VISIBLE_DEVICES=0 python examples/tacotron2/extract_duration.py \
  --rootdir ./tacotron2_dump/spk_01/valid/ \
  --outdir ./tacotron2_dump/spk_01/valid/durations/ \
  --checkpoint ./examples/tacotron2/exp/train.tacotron2.v1.spk_01/checkpoints/model-100000.h5 \
  --use-norm 1 \
  --config ./examples/tacotron2/conf/tacotron2.v1.yaml \
  --batch-size 32
  --win-front 3 \
  --win-back 3
```

Extract for training set:

```
CUDA_VISIBLE_DEVICES=0 python examples/tacotron2/extract_duration.py \
  --rootdir ./tacotron2_dump/spk_01/train/ \
  --outdir ./tacotron2_dump/spk_01/train/durations/ \
  --checkpoint ./examples/tacotron2/exp/train.tacotron2.v1.spk_01/checkpoints/model-100000.h5 \
  --use-norm 1 \
  --config ./examples/tacotron2/conf/tacotron2.v1.yaml \
  --batch-size 32
  --win-front 3 \
  --win-back 3
```

To extract postnets for training vocoder, follow above steps but with `extract_postnets.py`

## 4) Collating durations
After completing the extraction of the durations of the three speakers, the durations in the training and test sets of each speaker are collated together.

```
|- mntts2/
|   |- durations
|       |- 01_1_uttID-durations.npy
|       |- ......
|       |- 02_1_uttID-durations.npy
|       |- ......
|       |- 03_1_uttID-durations.npy
|       |- ......
|   |- train.txt
|   |- spk_01/
|       |- file1.wav
|       |- file1.txt
|       |- ......
|       |- spk_01_train.txt
|   |- spk_02/
|       |- file1.wav
|       |- file1.txt
|       |- ......
|       |- spk_02_train.txt
|   |- spk_03/
|       |- file1.wav
|       |- file1.txt
|       |- ......
|       |- spk_03_train.txt
```

## 5) Training FastSpeech2 from scratch with MnTTS dataset

Based on the script [`train_fastspeech2.py`](https://github.com/dathudeptrai/TensorFlowTTS/blob/master/examples/fastspeech2/train_fastspeech2.py).

First you need to refer to the instructions in 2) and modify the mntts.py file. Before you are ready to start fastspeech2,  you should modify mntts.py in  MNTTSProcessor class to look like the following code block:
```
    positions = {
        "speaker_name": 0,
        "wave_file": 1,
        "text": 2,
        "text_norm": 2, 
    }
    train_f_name: str = "train.txt"

    def create_items(self):
        if self.data_dir:
            with open(
                    os.path.join(self.data_dir, self.train_f_name), encoding="utf-8"
            ) as f:
                self.items = [self.split_line(self.data_dir, line, "|") for line in f]

    def split_line(self, data_dir, line, split):
        parts = line.strip().split(split)
        wave_file = parts[self.positions["wave_file"]]
        text_norm = parts[self.positions["text_norm"]]
        speaker_name = parts[self.positions["speaker_name"]]
        wav_path = os.path.join(data_dir,speaker_name,f"{wave_file}.wav")
        
        return text_norm, wav_path, speaker_name
```
After completing the above work, you are ready to start the FastSpeech2 work officially.


```
CUDA_VISIBLE_DEVICES=0 tensorflow-tts-preprocess \
  --rootdir ./mntts2 \
  --outdir ./fastspeech2_dump \
  --config ./preprocess/mntts_preprocess.yaml \
  --dataset mntts
```


```
CUDA_VISIBLE_DEVICES=0 tensorflow-tts-normalize \
  --rootdir ./fastspeech2_dump \
  --outdir ./fastspeech2_dump \
  --config ./preprocess/mntts_preprocess.yaml \
  --dataset mntts
```


Run fix mismatch to fix few frames difference in audio and duration files
```
CUDA_VISIBLE_DEVICES=0 python examples/mfa_extraction/fix_mismatch.py \
  --base_path ./fastspeech2_dump \
  --trimmed_dur_path ./mntts2/trimmed-durations \
  --dur_path ./mntts2/durations/
```


Checking the dimensional alignment of ids and durations.
```
python toolkit/fix.py
```


Change below example command line to match your dataset and run:
```bash
CUDA_VISIBLE_DEVICES=0 python examples/fastspeech2_mntts2/train_fastspeech2.py \
  --train-dir ./fastspeech2_dump/train/ \
  --dev-dir ./fastspeech2_dump/valid/ \
  --outdir ./examples/fastspeech2/exp/train.fastspeech2.v1/ \
  --config ./examples/fastspeech2/conf/fastspeech2.v1.yaml \
  --use-norm 1 \
  --f0-stat ./fastspeech2_dump/stats_f0.npy \
  --energy-stat ./fastspeech2_dump/stats_energy.npy \
  --mixed_precision 1 \
  --dataset_config ./preprocess/mntts_preprocess.yaml \
  --dataset_stats ./fastspeech2_dump/stats.npy \
  --dataset_mapping ./fastspeech2_dump/mntts_mapper.json
```


## 6) Vocoder Training For Each Speaker


Take speaker 01 for example.

First, you need training generator with only stft loss:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/hifigan/train_hifigan.py \
  --train-dir ./tacotron2_dump/spk_01/train/ \
  --dev-dir ./tacotron2_dump/spk_01/valid/ \
  --outdir ./examples/hifigan/exp/train.hifigan.v1.spk_01/ \
  --config ./examples/hifigan/conf/hifigan.v1.yaml \
  --use-norm 1 \
  --generator_mixed_precision 1 \
  --resume ""
```

Then resume and start training generator + discriminator:


```bash
CUDA_VISIBLE_DEVICES=0 python examples/hifigan/train_hifigan.py \
  --train-dir ./tacotron2_dump/spk_01/train/ \
  --dev-dir ./tacotron2_dump/spk_01/valid/ \
  --outdir ./examples/hifigan/exp/train.hifigan.v1.spk_01/ \
  --config ./examples/hifigan/conf/hifigan.v1.yaml \
  --use-norm 1 \
  --resume ./examples/hifigan/exp/train.hifigan.v1.spk_01/checkpoints/ckpt-100000
```

## 7) MnTTS Model Inference

You can follow below example command line to generate synthesized speech for a given text in 'prediction/spk_01/inference.txt' using Griffin-Lim and trained HiFi-GAN vocoder, take speaker 01 for example:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/fastspeech2_mntts2/mntts2_inference_fastspeech2.py \
    --outdir prediction/spk_01/MnTTS_inference \
    --infile prediction/spk_01/inference.txt  \
    --tts_ckpt examples/fastspeech2/exp/train.fastspeech2.v1/checkpoints/model-200000.h5 \
    --vocoder_ckpt  examples/hifigan/exp/train.hifigan.v1.spk_01/checkpoints/generator-200000.h5 \
    --stats_path fastspeech2_dump/stats.npy \
    --dataset_config preprocess/mntts_preprocess.yaml \
    --tts_config examples/fastspeech2/conf/fastspeech2.v1.yaml \
    --vocoder_config examples/hifigan/conf/hifigan.v1.yaml \
    --lan_json fastspeech2_dump/mntts_mapper.json 
    --speaker_id 0
```



## Links




Part of the synthesised audition audio：[DemoAudios]:(https://github.com/ssmlkl/MnTTS2/tree/master/DemoAudios)
[//]: # (Please kindly cite the following paper if you use this code repository in your work,)
[//]: # (```)
[//]: # (```)




## Acknowledgements:


Tensorflow-TTS: [https://github.com/TensorSpeech/TensorFlowTTS](https://github.com/TensorSpeech/TensorFlowTTS)

MnTTS: [https://github.com/walker-hyf/MnTTS](https://github.com/walker-hyf/MnTTS)
