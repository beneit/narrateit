import os
import json
import gc
import argparse
from narrateit.util import save_json, config, annotate_without_llm, read_file, convert_pdf_to_txt
from narrateit.narrate import resolve_characters


def get_user_info(s):
    while True:
        if s == 'use_characters':
            user_input = input("Use separate voices for different characters? [y/n]\n")
            if user_input.lower() in ['y', 'n']:
                return user_input.lower() == 'y'
            print("Invalid Input, answer y or n.")
        elif s == 'load_annotations':
            user_input = input("Load annotations from previous output? [y/n]\n")
            if user_input.lower() in ['y', 'n']:
                return user_input.lower() == 'y'
            print("Invalid Input, answer y or n.")
        else:
            raise ValueError(f"Invalid user_info: {s}")


def annotate():
    from narrateit.annotate import clean_string
    from narrateit.prompt_util import extract_characters

    # Create annotations
    text = clean_string(read_file(config['document']))
    use_characters = get_user_info("use_characters")
    if use_characters:
        load_annotations = get_user_info("load_annotations")
        if load_annotations:
            annotation_file = os.path.join('output', config['_document_name'], 'annotations.json')
            with open(annotation_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            from narrateit.annotate import get_model
            model = get_model()
            annotations = model.annotate(text)
            # TODO: Do characters via separate model that probably uses the same LLM in the background
            characters = extract_characters(annotations)
            del model
            import torch
            torch.cuda.empty_cache()
            gc.collect()
            return {'annotation': annotations,
                    'characters': characters,
                    'characters_resolved': False,
                    'has_voice_mappings': False}
    else:
        return annotate_without_llm(text)


def narrate(document):
    from narrateit.tts.xtts_util import create_wave_files, load_tts
    from narrateit.narrate import merge_audio
    # Create wave files
    model, tokenizer = load_tts(config['tts_model'])
    output_path = os.path.join('output', config['_document_name'], 'waves')
    create_wave_files(document, model, tokenizer, output_path)
    # Create final audio
    return merge_audio(document)


def main():
    parser = argparse.ArgumentParser(description='Convert text or pdf to an audiobook. In case of no arguments will run'
                                                 ' complete narration according to the config file')
    parser.add_argument('--convert_pdf', action='store_true', help='Convert a pdf to text file')
    parser.add_argument('--annotate', action='store_true', help='Annotate according to what character or narration '
                                                                'speaks which part of the text')
    parser.add_argument('--characters', action='store_true', help='Resolve the annotated characters, correct '
                                                                  'mistakes/remove dublicates')
    parser.add_argument('--narrate', action='store_true', help='Generate the audio')
    
    args = parser.parse_args()
    
    # Convert to pdf
    if args.convert_pdf:
        path = input("Input path to pdf file:\n")
        path = os.path.normpath(path)
        text = convert_pdf_to_txt(path)
        file_name = 'books' + os.path.sep + '.'.join(os.path.basename(path).split('.')[:-1]) + '.txt'
        with open(file_name, "w", encoding="utf-8") as file:
            file.write(text)
        other_args = {k: v for k, v in vars(args).items() if k != 'convert_pdf'}
        add_to_config = any(other_args.values()) or \
            input(f"Text written to '{file_name}'. Add file to config.json [y/n]? ") == 'y'
        if add_to_config:
            config['document'] = file_name
            save_json(os.path.join("config", "config.json"), config)
    
    # Create output structure
    if not os.path.exists('output'):
        os.mkdir('output')
    config['_document_name'] = '.'.join((os.path.split(config['document'])[-1]).split('.')[:-1])
    annotation_file = os.path.join('output', config['_document_name'], 'annotations.json')
    document_folder = os.path.join('output', config['_document_name'])
    os.makedirs(os.path.join('output', config['_document_name'], 'waves'), exist_ok=True)
    
    # Only annotate
    if args.annotate:
        document = annotate()
        save_json(annotation_file, document)
    
    # Only characterize
    if args.characters:
        with open(annotation_file, 'r', encoding='utf-8') as f:
            document = json.load(f)
        document = resolve_characters(document)
        save_json(annotation_file, document)
    
    # Only narrate
    output_format = 'mp3'
    audio_file = os.path.join('output', config['_document_name'], config['_document_name']) + '.' + output_format
    if args.narrate:
        with open(annotation_file, 'r', encoding='utf-8') as f:
            document = json.load(f)
        if not document.get('characters_resolved', False):
            print("Characters have not been resolved yet, do so now:")
            document = resolve_characters(document)
        merged_audio = narrate(document)
        save_json(annotation_file, document)
        merged_audio.export(audio_file, format=output_format)
    
    # Do the standard pipeline in case no keywords are given
    if not any(vars(args).values()):
        document = annotate()
        save_json(annotation_file, document)
        document = resolve_characters(document)
        save_json(annotation_file, document)
        merged_audio = narrate(document)
        merged_audio.export(audio_file, format=output_format)


def train_peft():
    from narrateit.annotate import read_file, get_prompt
    from narrateit.transformer_util import load_model_tokenizer
    from narrateit.peft_util import create_peft_model, train_peft
    
    # Define Hyperparameter and datasets
    num_virtual_tokens = 10
    num_epochs = 2
    learning_rate = 0.05
    dataset = "datasets/annotation"
    
    # load language model
    model_path = config['annotation_model']
    model, tokenizer = load_model_tokenizer(model_path)
    replacements = {"DOCUMENT_TYPE": config['document_type'], "EOS": tokenizer.eos_token}
    prompt = get_prompt(read_file(config['system_prompt']), {}, [], [], replacements)
    print(prompt)
    peft_model = create_peft_model(model, num_virtual_tokens=num_virtual_tokens, initial_prompt=prompt)
    train_peft(peft_model, dataset, num_epochs=num_epochs, learning_rate=learning_rate)


if __name__ == "__main__":
    main()
