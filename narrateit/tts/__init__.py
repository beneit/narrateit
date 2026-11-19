"""
TTS engine selection. The engine is derived from the folder name of config['tts_model'], e.g.
'models_tts/XTTS-v2' -> xtts_util, 'models_tts/Kokoro-82M' -> kokoro_util.
Engine modules are only imported when requested since their imports are expensive.
Every engine module provides load_tts(model_path) and create_wave_files(document, model, tts_config, output_path).
"""
import os
import importlib

ENGINES = {
    'xtts': 'narrateit.tts.xtts_util',
    'kokoro': 'narrateit.tts.kokoro_util',
}


def get_engine_name(model_path):
    folder = os.path.basename(os.path.normpath(model_path.replace('\\', '/'))).lower()
    for name in ENGINES:
        if folder.startswith(name):
            return name
    raise ValueError(f"No TTS engine known for model '{model_path}'. Known engines: {', '.join(ENGINES)}")


def import_tts(model_path):
    """Returns (load_tts, create_wave_files) of the engine matching model_path."""
    module = importlib.import_module(ENGINES[get_engine_name(model_path)])
    return module.load_tts, module.create_wave_files
