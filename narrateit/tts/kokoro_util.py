"""
Kokoro-82M backend for narrateit.

Interface (shared with xtts_util, selected via narrateit.tts.import_tts):
    load_tts(model_path)                                     -> model, tts_config
    create_wave_files(document, model, tts_config, output_path)

Kokoro does its own text chunking, so every annotation produces exactly one wave file.
Only American English voices are used for now (lang_code 'a').
"""
import os
import glob
import math
import itertools
import warnings
from collections import Counter

import numpy as np
import torch
import soundfile as sf
import pyloudnorm as pyln
from tqdm import tqdm
from kokoro import KModel, KPipeline

ENGINE = 'kokoro'
REPO_ID = 'hexgrad/Kokoro-82M'  # only used to silence kokoro's warning; weights/config/voices are loaded locally
LANG_CODE = 'a'                 # American English
SAMPLE_RATE = 24000
TARGET_LOUDNESS = -14.0         # LUFS, same as xtts_util
MIN_BLEND_SCORE = 2             # presets graded below D are not used as blend parents

device = "cuda" if torch.cuda.is_available() else "cpu"
warnings.filterwarnings("ignore", category=UserWarning, module="pyloudnorm")

# Overall grades from models_tts/Kokoro-82M/VOICES.md (American English)
PRESET_GRADES = {
    'af_heart': 'A', 'af_bella': 'A-', 'af_nicole': 'B-', 'af_aoede': 'C+', 'af_kore': 'C+', 'af_sarah': 'C+',
    'af_alloy': 'C', 'af_nova': 'C', 'af_sky': 'C-', 'af_jessica': 'D', 'af_river': 'D',
    'am_fenrir': 'C+', 'am_michael': 'C+', 'am_puck': 'C+', 'am_echo': 'D', 'am_eric': 'D', 'am_liam': 'D',
    'am_onyx': 'D', 'am_santa': 'D-', 'am_adam': 'F+',
}
GRADE_SCORES = {'A': 10, 'A-': 9, 'B+': 8.5, 'B': 8, 'B-': 7, 'C+': 6, 'C': 5, 'C-': 4,
                'D+': 3, 'D': 2, 'D-': 1, 'F+': 0.5, 'F': 0}

# Same human-readable names as the XTTS voices (scripts/xtts_generate_voice_samples.py)
NAMES_MALE = ['Gary', 'Jon', 'Rick', 'David', 'Jordan', 'Mike', 'Yann', 'James', 'Eric', 'Will', 'Jason', 'Aaron',
              'Patrick', 'Jerry', 'Bill', 'Tom', 'Bruce', 'Henry', 'Oscar', 'Leo', 'Samuel', 'Victor', 'Frank',
              'Martin', 'Peter', 'Hugo', 'Simon']
NAMES_FEMALE = ['Laura', 'Lea', 'Karen', 'Brenda', 'Eileen', 'Joy', 'Lauren', 'Rose', 'Naomie', 'Alisa', 'Tina',
                'Jenna', 'Carol', 'Barbara', 'Rebecca', 'Anna', 'Emily', 'Clara', 'Sophie', 'Grace', 'Helen', 'Iris',
                'Julia', 'Maya', 'Nora', 'Olivia', 'Ruth']


def _build_catalog():
    """
    Deterministic list of voices: {'id', 'gender', 'quality', 'presets'}.
    Per gender: presets ordered by grade, followed by 50/50 blends of two presets (best pairs first, each preset
    used in a limited number of blends so they stay distinct) until every name in the name list is used, and finally
    the presets graded below MIN_BLEND_SCORE.
    The catalog order is the order in which voices are handed out.
    """
    catalog = []
    for gender, prefix, names in (('f', 'af_', NAMES_FEMALE), ('m', 'am_', NAMES_MALE)):
        presets = [(p, GRADE_SCORES[g]) for p, g in PRESET_GRADES.items() if p.startswith(prefix)]
        presets.sort(key=lambda x: -x[1])
        n_blends = max(len(names) - len(presets), 0)
        parents = [x for x in presets if x[1] >= MIN_BLEND_SCORE]
        pairs = sorted(itertools.combinations(parents, 2), key=lambda ab: -(ab[0][1] + ab[1][1]))
        cap = math.ceil(2 * n_blends / len(parents)) if parents else 0
        count = Counter()
        chosen = []
        for a, b in pairs:
            if len(chosen) >= n_blends:
                break
            if count[a[0]] < cap and count[b[0]] < cap:
                chosen.append((a, b))
                count[a[0]] += 1
                count[b[0]] += 1
        for pair in pairs:  # fill up in case the cap was too strict
            if len(chosen) >= n_blends:
                break
            if pair not in chosen:
                chosen.append(pair)
        blends = [{'presets': [a[0], b[0]], 'quality': (a[1] + b[1]) / 2} for a, b in chosen]
        blends.sort(key=lambda x: -x['quality'])
        # good presets first, then blends, poor presets (e.g. am_adam, am_santa) only as a last resort
        entries = [{'presets': [p], 'quality': q} for p, q in presets if q >= MIN_BLEND_SCORE] + blends + \
                  [{'presets': [p], 'quality': q} for p, q in presets if q < MIN_BLEND_SCORE]

        for name, entry in zip(names, entries):
            catalog.append({'id': name, 'gender': gender, **entry})
    return catalog


CATALOG = _build_catalog()
VOICE_BY_ID = {v['id']: v for v in CATALOG}


def load_tts(model_path):
    print(f"Loading model '{model_path}'...")
    weights = sorted(glob.glob(os.path.join(model_path, '*.pth')))
    if not weights:
        raise FileNotFoundError(f"No .pth weights found in '{model_path}'")
    model = KModel(repo_id=REPO_ID, config=os.path.join(model_path, 'config.json'), model=weights[0])
    model = model.to(device).eval()
    pipeline = KPipeline(lang_code=LANG_CODE, repo_id=REPO_ID, model=model)
    tts_config = {'pipeline': pipeline,
                  'voices_dir': os.path.join(model_path, 'voices'),
                  'sample_rate': SAMPLE_RATE,
                  'voice_cache': {}}
    return model, tts_config


def load_voice(voice_id, tts_config):
    """Returns the style tensor for a catalog voice (blends are the mean of their presets)."""
    cache = tts_config['voice_cache']
    if voice_id not in cache:
        presets = VOICE_BY_ID[voice_id]['presets']
        packs = [torch.load(os.path.join(tts_config['voices_dir'], f'{p}.pt'), map_location='cpu', weights_only=True)
                 for p in presets]
        cache[voice_id] = torch.stack(packs).mean(dim=0).float()
    return cache[voice_id]


def _normalize(audio, sample_rate):
    meter = pyln.Meter(sample_rate)
    # integrated loudness needs at least one 400ms block; measure very short lines on a repeated copy
    measure = audio
    min_len = int(0.5 * sample_rate)
    if 0 < len(measure) < min_len:
        measure = np.tile(measure, math.ceil(min_len / len(measure)))
    loudness = meter.integrated_loudness(measure) if len(measure) >= min_len else -np.inf
    if not np.isfinite(loudness):
        return audio
    audio = pyln.normalize.loudness(audio, loudness, TARGET_LOUDNESS)
    peak = np.max(np.abs(audio))
    if peak > 1.0:  # avoid clipping when writing 16-bit wav
        audio = audio / peak * 0.99
    return audio


def create_wave(prompt, voice, tts_config, output_file=None, speed=1.0):
    """voice: a voice_id from the catalog or a style tensor. Returns the duration in seconds."""
    if output_file is None:
        output_file = "kokoro_tts_out.wav"
    if isinstance(voice, str):
        voice = load_voice(voice, tts_config)
    sample_rate = tts_config['sample_rate']
    chunks = [audio for _, _, audio in tts_config['pipeline'](prompt, voice=voice, speed=speed)
              if audio is not None]
    if chunks:
        audio = torch.cat([torch.as_tensor(c).flatten().cpu() for c in chunks]).numpy().astype(np.float64)
        audio = _normalize(audio, sample_rate)
    else:  # nothing speakable (e.g. only punctuation): keep a short silence so the merge still works
        audio = np.zeros(int(0.25 * sample_rate))
    sf.write(output_file, audio, sample_rate)
    return len(audio) / sample_rate


def _gender(character):
    gender = str(character.get('gender', '')).lower()[:1]
    if gender not in ('m', 'f'):
        raise ValueError(f"Character '{character.get('name')}' has no gender (m/f), resolve characters first.")
    return gender


def map_speakers(document):
    """
    Sets 'voice_id' on every character. Existing Kokoro mappings (document['tts_engine'] == 'kokoro') are kept as long
    as the voice_id exists, so mappings can be edited by hand. All other characters get the best unused voice of their
    gender, most dialogue first. If a gender runs out of voices, the least used voice is reused.
    """
    characters = document['characters']
    keep_existing = document.get('tts_engine') == ENGINE
    usage = Counter()
    pending = []
    for character in sorted(characters.values(), key=lambda c: c.get('n_dialogue', 0), reverse=True):
        character.pop('voice_sample', None)  # XTTS leftover, not used
        voice_id = character.get('voice_id') if keep_existing else None
        if voice_id in VOICE_BY_ID:
            usage[voice_id] += 1
        else:
            pending.append(character)
    for character in pending:
        candidates = [v for v in CATALOG if v['gender'] == _gender(character)]
        voice = min(candidates, key=lambda v: usage[v['id']])  # ties resolve in catalog (= quality) order
        if usage[voice['id']] > 0:
            print(f"Warning: out of unused voices, '{character.get('name')}' shares voice '{voice['id']}'")
        character['voice_id'] = voice['id']
        usage[voice['id']] += 1


def create_wave_files(document, model, tts_config, output_path):
    os.makedirs(output_path, exist_ok=True)
    characters = document['characters']
    map_speakers(document)
    document['has_voice_mappings'] = True
    document['tts_engine'] = ENGINE
    print("Voice Mappings:")
    for c in sorted(characters.values(), key=lambda x: x.get('n_dialogue', 0), reverse=True):
        print(f"{c['name']}: {c['voice_id']} ({' + '.join(VOICE_BY_ID[c['voice_id']]['presets'])})")
    voices = {key: load_voice(c['voice_id'], tts_config) for key, c in characters.items()}

    annotations = document['annotation']
    for i, annotation in enumerate(tqdm(annotations)):
        output_file = os.path.join(output_path, f'out_{i:04}_00.wav')
        annotation['duration'] = create_wave(annotation['text'], voices[annotation['speaker']], tts_config,
                                             output_file=output_file)
        annotation['waves'] = [output_file]
    return len(annotations)


if __name__ == "__main__":
    model, tts_config = load_tts(os.path.join('models_tts', 'Kokoro-82M'))
    create_wave("Mum said I never said HE stole the money! Also, can you help me with my Kakuroro?", 'Laura', tts_config, output_file='kokoro_tts_out.wav')
