import os
import json
from narrateit.util import save_json, config



def main():
    from narrateit.tts_parler_util import create_wave_files, load_tts
    from narrateit.narrate import merge_audio
    # TODO Move imports to top
    # Create wave files
    with open(config['annotation_file'], 'r', encoding='utf-8') as f:
        document = json.load(f)
    model, tokenizer = load_tts(config['tts_model'])
    create_wave_files(document, model, tokenizer)
    # Create final audio
    merged_audio = merge_audio()
    output_format = 'mp3'
    output_file = '.'.join(os.path.split(config['document'])[1].split('.')[:-1] + [output_format])
    merged_audio.export(os.path.join('output', output_file), format=output_format)


if __name__ == "__main__":
    # from narrateit.narrate import resolve_characters
    from narrateit.tts_parler_util import create_wave_files, load_tts
    from narrateit.narrate import merge_audio
    
    with open(config['annotation_file'], 'r', encoding='utf-8') as f:
        document = json.load(f)
    model, tokenizer = load_tts(config['tts_model'])
    create_wave_files(document, model, tokenizer)
    merged_audio = merge_audio()
    output_format = 'mp3'
    output_file = '.'.join(os.path.split(config['document'])[1].split('.')[:-1] + [output_format])
    merged_audio.export(os.path.join('output', output_file), format=output_format)
