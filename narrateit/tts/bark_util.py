import os
import re

import torch
from transformers import AutoModel, AutoProcessor
import soundfile as sf
import pyloudnorm as pyln
from tqdm import tqdm
import warnings

from narrateit.util import config


device = "cuda" if torch.cuda.is_available() else "cpu"


def load_tts(model_path):
    print(f"Loading model '{model_path}'...")
    model = AutoModel.from_pretrained(model_path).to(device)
    model.config.sampling_rate = 24000
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


# TODO: Return raw data instead
def create_wave(prompt, voice, model, processor, output_file=None):
    if output_file is None:
        output_file = "bark_tts_out.wav"
    inputs = processor(prompt, voice_preset=voice)
    generation = model.generate(**inputs.to(device))
    audio = generation.cpu().numpy().squeeze()
    meter = pyln.Meter(model.config.sampling_rate)
    # audio = pyln.normalize.peak(audio, -1.0)
    loudness = meter.integrated_loudness(audio)
    audio = pyln.normalize.loudness(audio, loudness, -14.0)
    sf.write(output_file, audio, model.config.sampling_rate)


if __name__ == "__main__":
    model, tokenizer = load_tts('tts/bark-small')
    prompt = "Hello, my name is Suno. And, uh — and I like pizza. [laughs] But I also have other interests such as playing tic tac toe."
    voice_preset = "en_speaker_6" #tts/bark-small/speaker_embeddings/v2/
    create_wave(prompt, voice_preset, model, tokenizer, output_file=None)
