import os
import re
import json

import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import soundfile as sf
import pyloudnorm as pyln
from tqdm import tqdm
import warnings

from narrateit.util import config

voices = json.load(open(os.path.join('output', 'xtts_voices', 'voices.json')))
device = "cuda" if torch.cuda.is_available() else "cpu"
warnings.filterwarnings("ignore", category=UserWarning, module="pyloudnorm")


def map_speakers(characters, model):
    """
    Adds voice_id, voice_sample variables to each character. Returns voice_latents which is a dict with character names
    as keys and [gpt_cond_latent, speaker_embedding] as values
    """
    # Sort characters by n_dialogue in descending order
    sorted_characters = sorted(characters.items(), key=lambda x: x[1]['n_dialogue'], reverse=True)
    # Sort voices by quality in descending order
    sorted_voices = sorted(voices, key=lambda x: x['quality'], reverse=True)
    used_samples = set()
    voice_latents = {}
    for character_key, character in sorted_characters:
        character['voice_sample'] = None
        for voice in sorted_voices:
            if voice['sample'] in used_samples:
                continue
            if character['gender'].lower() == voice['gender'].lower() and abs(character['age'] - voice['age']) <= 10:
                character.update({'voice_id': voice['id'], 'voice_sample': voice['sample']})
                gptcl, se = model.get_conditioning_latents(audio_path=voice['sample'], gpt_cond_len=30,
                                                           gpt_cond_chunk_len=6)
                voice_latents[character_key] = [gptcl, se]
                used_samples.add(voice['sample'])
                break
        if character['voice_sample'] is not None:
            continue
        # Do another run without age restriction just in case we couldn't match because of it
        for voice in sorted_voices:
            if voice['sample'] in used_samples:
                continue
            if character['gender'].lower() == voice['gender'].lower():
                character.update({'voice_id': voice['id'], 'voice_sample': voice['sample']})
                gptcl, se = model.get_conditioning_latents(audio_path=voice['sample'], gpt_cond_len=30,
                                                           gpt_cond_chunk_len=6)
                voice_latents[character_key] = [gptcl, se]
                used_samples.add(voice['sample'])
                break
        if character['voice_sample'] is None:
            raise ValueError(f"No match found for character: {character}")
    return voice_latents


def load_tts(model_path):
    print(f"Loading model '{model_path}'...")
    xtts_config = XttsConfig()
    xtts_config.load_json(os.path.join(model_path, "config.json"))
    model = Xtts(xtts_config)
    model.load_checkpoint(xtts_config, checkpoint_dir=model_path, eval=True)
    if device[:4] == 'cuda':
        model.cuda()
    return model, xtts_config


def make_embeddings(speaker_wavs, model):
    gpt_cond_latents = []
    speaker_embeddings = []
    for speaker_wav in speaker_wavs:
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=speaker_wav)
        gpt_cond_latents.append(gpt_cond_latent)
        speaker_embeddings.append(speaker_embedding)
    return gpt_cond_latents, speaker_embeddings


def create_wave(prompt, model, tts_config,
                speaker_wav=None, gpt_cond_latent=None, speaker_embedding=None, output_file=None):
    if output_file is None:
        output_file = "xtts2_tts_out.wav"
    if speaker_wav is None:
        generation = model.inference(text=prompt, gpt_cond_latent=gpt_cond_latent, speaker_embedding=speaker_embedding,
                                     language="en", enable_text_splitting=True)
    else:
        generation = model.synthesize(prompt, tts_config,
                                      speaker_wav=speaker_wav, gpt_cond_len=30, language="en")
    audio = generation['wav']
    meter = pyln.Meter(tts_config.audio.output_sample_rate)
    # audio = pyln.normalize.peak(audio, -1.0)
    loudness = meter.integrated_loudness(audio)
    audio = pyln.normalize.loudness(audio, loudness, -14.0)
    sf.write(output_file, audio, tts_config.audio.output_sample_rate)
    return sf.info(output_file).duration


def create_wave_files(document, model, tts_config, output_path):
    def segment(text):
        segments = []
        part = ''
        length = 0
        min_segment_length = 200
        sentences = re.split(r'(?<=[.!?])', text)
        for sentence in sentences:
            length += len(sentence)
            part += sentence
            if length > min_segment_length:
                segments.append(part)
                part = ''
                length = 0
        if length > 0:
            segments.append(part)
        return segments

    os.makedirs(output_path, exist_ok=True)
    characters = document['characters']
    voice_latents = map_speakers(characters, model)
    document['has_voice_mappings'] = True
    print("Voice Mappings:")
    for c in sorted(characters.values(), key=lambda x: x['n_dialogue'], reverse=True):
        print(f"{c['name']}: {c['voice_sample']}")
    annotations = document['annotation']
    wave_number_1 = 0
    wave_number = 0
    for annotation in tqdm(annotations):
        text = annotation['text']
        gpt_cond_latent, speaker_embedding = voice_latents[annotation['speaker']]
        if len(text) > 300:
            text = segment(text)
        else:
            text = [text]
        wave_number_2 = 0
        duration = 0
        wave_files = []
        for prompt in tqdm(text):
            output_file = os.path.join(output_path, f'out_{wave_number_1:04}_{wave_number_2:02}.wav')
            dt = create_wave(prompt, model, tts_config, gpt_cond_latent=gpt_cond_latent,
                        speaker_embedding=speaker_embedding, output_file=output_file)
            wave_number += 1
            wave_number_2 += 1
            duration += dt
            wave_files.append(output_file)
        wave_number_1 += 1
        annotation['duration'] = duration
        annotation['waves'] = wave_files
    return wave_number


if __name__ == "__main__":
    model, tts_config = load_tts('model_tts/XTTS-v2')
    speaker = "model_tts/XTTS-v2/samples/LJ001-0026.wav"
    # prompt = "The quick brown fox jumps over the lazy dog! This sentence contains every letter of the alphabet, making it ideal for testing pronunciation and clarity."
    prompt = 'I never said HE stole the money!'
    for voice in tqdm(voices):
        speaker_wav = voice["sample"]
        output = os.path.join("output", "waves", "temp", os.path.split(speaker_wav)[1])
        create_wave(prompt, model, tts_config, speaker_wav, output_file=output)
