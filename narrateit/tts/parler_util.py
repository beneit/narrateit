import os
import re

import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import pyloudnorm as pyln
from tqdm import tqdm
import warnings

from narrateit.util import config


device = "cuda" if torch.cuda.is_available() else "cpu"
warnings.filterwarnings("ignore", category=UserWarning, module="pyloudnorm")


def load_tts(model_path):
    print(f"Loading model '{model_path}'...")
    model = ParlerTTSForConditionalGeneration.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


# TODO: Return raw data instead
def create_wave(prompt, description, model, tokenizer, output_file=None):
    if output_file is None:
        output_file = "parler_tts_out.wav"
    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    audio = generation.cpu().numpy().squeeze()
    meter = pyln.Meter(model.config.sampling_rate)
    # audio = pyln.normalize.peak(audio, -1.0)
    loudness = meter.integrated_loudness(audio)
    audio = pyln.normalize.loudness(audio, loudness, -14.0)
    sf.write(output_file, audio, model.config.sampling_rate)


# TODO: Return raw data instead
def create_wave_files(document: dict, model, tokenizer):
    speech_config = {"quality": ["very clear audio"],
                     "speech_monotony": ["slightly expressive and animated", "monotone"],
                     "reverberation": ["slightly close sounding", "very distant sounding"],
                     "speaking_rate": ["moderate speed", "slightly fast", "slowly"],
                     "accent": ["English", "American", "Irish", "Unidentified", "Canadian", "Australian", "Scottish"]}
    
    voices_male = ['Gary', 'Jon', 'Rick', 'David', 'Jordan', 'Mike', 'Yann', 'James', 'Eric', 'Will', 'Jason', 'Aaron',
                   'Patrick', 'Jerry', 'Bill', 'Tom', 'Rebecca', 'Bruce']
    voices_female = ['Laura', 'Lea', 'Karen', 'Brenda', 'Eileen', 'Joy', 'Lauren', 'Rose', 'Naomie', 'Alisa', 'Tina',
                     'Jenna', 'Carol', 'Barbara', 'Rebecca', 'Anna', 'Emily']
    
    def map_speakers(characters):
        narrator = "Bruce"
        # if characters['Narration']['name'] != 'nameless':
        if characters['Narration']['gender'] == 'm':
            characters['Narration']['voice'] = 'Bruce'
        else:
            characters['Narration']['voice'] = 'Brenda'
        for name in characters:
            if name == "Narration":
                continue
            elif name == "Harry":
                characters[name]['voice'] = "David"
            elif name == "Michael":
                characters[name]['voice'] = "Will"
            elif name == "Petunia":
                characters[name]['voice'] = "Naomi"
            elif name == "Mrs. Figg":
                characters[name]['voice'] = "Brenda"
            else:
                voices = voices_male if characters[name]['gender'] == 'm' else voices_female
                characters[name]['voice'] = ''
                for a_voice in voices:
                    used_voices = [characters[n].get('voice', '') for n in characters]
                    if a_voice not in used_voices:
                        characters[name]['voice'] = a_voice
                        break
        return characters
    
    def segment(text):
        segments = []
        part = ''
        length = 0
        min_segment_length = 300
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
    
    characters = document['characters']
    characters = map_speakers(characters)
    annotations = document['annotation']
    wave_number_1 = 0
    wave_number = 0
    for annotation in tqdm(annotations):
        text = annotation['text']
        if len(text) > 400:
            text = segment(text)
        else:
            text = [text]
        age = characters[annotation['speaker']]['age']
        name = characters[annotation['speaker']]['voice']
        description = f"{age} year old {name}'s english accent is slightly expressive and animated, with very clear audio"
        wave_number_2 = 0
        for prompt in text:
            output_file = os.path.join('output', 'waves', f'out_{wave_number_1:04}_{wave_number_2:02}.wav')
            # skip the actual synthesis for now if the file already exists # TODO delete this
            if not os.path.isfile(output_file):
                create_wave(prompt, description, model, tokenizer, output_file=output_file)
            wave_number += 1
            wave_number_2 += 1
        wave_number_1 += 1
    return wave_number


def test_voices(prompt, gender, age, model, tokenizer):
    voices_male = ['Gary', 'Jon', 'Rick', 'David', 'Jordan', 'Mike', 'Yann', 'James', 'Eric', 'Will', 'Jason', 'Aaron',
                   'Patrick', 'Jerry', 'Bill', 'Tom', 'Bruce']
    voices_female = ['Laura', 'Lea', 'Karen', 'Brenda', 'Eileen', 'Joy', 'Lauren', 'Rose', 'Naomie', 'Alisa', 'Tina',
                     'Jenna', 'Carol', 'Barbara', 'Rebecca', 'Anna', 'Emily']
    voices = voices_male if gender == 'm' else voices_female
    for name in tqdm(voices):
        description = f"{age} year old {name}'s english accent is slightly expressive and animated, with very clear audio"
        output_file = f"output/waves/{name}_{age}_english.wav"
        create_wave(prompt, description, model, tokenizer, output_file=output_file)


if __name__ == "__main__":
    model, tokenizer = load_tts(config['tts_model'])
    prompt = "Dear, I understand that you're not familiar with the skeptical literature. You may not realize how easy it is for a trained magician to fake the seemingly impossible."
    " If it seemed like they could always guess what you were thinking, that's called cold reading—"
    description = "Jon's English accent is slightly expressive and animated, with very clear audio"
    # create_wave(prompt, description, model, tokenizer, output_file='output/waves/jon_english.wav')
    gender = 'm'
    age = 40
    test_voices(prompt, gender, age, model, tokenizer)
