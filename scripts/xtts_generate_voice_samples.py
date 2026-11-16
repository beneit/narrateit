import os
import sys
import json

# allow running as `python scripts/xtts_generate_voice_samples.py` from the main tree
sys.path.insert(0, os.getcwd())

import tempfile

import soundfile as sf
from tqdm import tqdm

from narrateit.tts.xtts_util import create_wave, load_tts

model_path = os.path.join('models_tts', 'XTTS-v2')
voices = json.load(open(os.path.join(model_path, 'samples', 'voices.json')))
voiceline = "This is my voice. Shall I read your book?"
output_waves = "output/xtts_voices"

voices_male = ['Gary', 'Jon', 'Rick', 'David', 'Jordan', 'Mike', 'Yann', 'James', 'Eric', 'Will', 'Jason', 'Aaron',
               'Patrick', 'Jerry', 'Bill', 'Tom', 'Bruce', 'Henry', 'Oscar', 'Leo', 'Samuel', 'Victor', 'Frank',
               'Martin', 'Peter', 'Hugo', 'Simon']
voices_female = ['Laura', 'Lea', 'Karen', 'Brenda', 'Eileen', 'Joy', 'Lauren', 'Rose', 'Naomie', 'Alisa', 'Tina',
                 'Jenna', 'Carol', 'Barbara', 'Rebecca', 'Anna', 'Emily', 'Clara', 'Sophie', 'Grace', 'Helen', 'Iris',
                 'Julia', 'Maya', 'Nora', 'Olivia', 'Ruth']


def resolve_sample(path):
    """voices.json stores paths like 'tts/XTTS-v2/samples/x.wav'; on disk they live under 'models_tts/...'."""
    candidates = [path,
                  'models_' + path,
                  os.path.join(model_path, 'samples', os.path.basename(path))]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"Could not find voice sample '{path}' (tried: {candidates})")


def assign_ids(voices):
    """Give every voice a unique name from the gendered name lists."""
    pools = {'m': list(voices_male), 'f': list(voices_female)}
    used = set()
    result = []
    for voice in voices:
        gender = voice['gender'].lower()[0]
        pool = pools[gender]
        while pool and pool[0] in used:
            pool.pop(0)
        if pool:
            name = pool.pop(0)
        else:  # ran out of names: fall back to a numbered name
            n = 1
            while f"Voice{gender.upper()}{n}" in used:
                n += 1
            name = f"Voice{gender.upper()}{n}"
            print(f"Warning: ran out of {gender} names, using '{name}'")
        used.add(name)
        result.append({**voice, 'id': name})
    return result


def wav_to_mp3(wav_file, mp3_file):
    """Convert with soundfile (needs libsndfile >= 1.1, bundled with soundfile >= 0.12); fall back to pydub/ffmpeg."""
    try:
        audio, sample_rate = sf.read(wav_file)
        sf.write(mp3_file, audio, sample_rate, format='MP3')
    except Exception as e:
        print(f"soundfile could not write mp3 ({e}), falling back to pydub")
        from pydub import AudioSegment
        AudioSegment.from_wav(wav_file).export(mp3_file, format='mp3', bitrate='128k')


os.makedirs(output_waves, exist_ok=True)
voices_out = assign_ids(voices)

# Create wave files
print("Loading voice model...")
model, tts_config = load_tts(model_path)
print("Model loaded successfully.")
for voice in tqdm(voices_out):
    sample = resolve_sample(voice['sample'])
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=sample, gpt_cond_len=30, gpt_cond_chunk_len=6)  # same settings as map_speakers
    # lowercase file names to match the web manifest (e.g. voices/laura.mp3)
    output_file = os.path.join(output_waves, f"{voice['id'].lower()}.mp3")
    with tempfile.TemporaryDirectory() as tmp:
        wav_file = os.path.join(tmp, 'voice.wav')
        create_wave(voiceline, model, tts_config, gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding, output_file=wav_file)
        wav_to_mp3(wav_file, output_file)

with open(os.path.join(output_waves, 'voices.json'), 'w') as f:
    json.dump(voices_out, f, indent=2)
print(f"Wrote {len(voices_out)} voice samples and voices.json to '{output_waves}'")
