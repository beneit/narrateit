import os
import sys
import json

# allow running as `python scripts/kokoro_generate_voice_samples.py` from the main tree
sys.path.insert(0, os.getcwd())

import tempfile

import soundfile as sf
from tqdm import tqdm

from narrateit.tts.kokoro_util import CATALOG, create_wave, load_tts

model_path = os.path.join('models_tts', 'Kokoro-82M')
voiceline = "This is my voice. Shall I read your book?"
output_waves = "output/kokoro_voices"


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

print("Loading voice model...")
model, tts_config = load_tts(model_path)
print("Model loaded successfully.")
for voice in tqdm(CATALOG):
    # lowercase file names to match the web manifest (e.g. voices/laura.mp3)
    output_file = os.path.join(output_waves, f"{voice['id'].lower()}.mp3")
    with tempfile.TemporaryDirectory() as tmp:
        wav_file = os.path.join(tmp, 'voice.wav')
        create_wave(voiceline, voice['id'], tts_config, output_file=wav_file)
        wav_to_mp3(wav_file, output_file)

with open(os.path.join(output_waves, 'voices.json'), 'w') as f:
    json.dump(CATALOG, f, indent=2)
print(f"Wrote {len(CATALOG)} voice samples and voices.json to '{output_waves}'")
