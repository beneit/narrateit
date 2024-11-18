import os
import json
import gc
import argparse
from narrateit.util import save_json, config, annotate_without_llm, read_file
from narrateit.narrate import resolve_characters


def get_user_info(s):
    while True:
        if s == 'use_characters':
            user_input = input("Use separate voices for different characters? [y/n]\n")
            if user_input.lower() in ['y', 'n']:
                return user_input.lower() == 'y'
            print("Invalid Input, answer y or n.")
        elif s == 'load_annotations':
            user_input = input("Load annotations from file? [y/n]\n")
            if user_input.lower() in ['y', 'n']:
                return user_input.lower() == 'y'
            print("Invalid Input, answer y or n.")
        else:
            raise ValueError(f"Invalid user_info: {s}")


def annotate():
    from narrateit.annotate import clean_string

    if not os.path.exists('output'):
        os.mkdir('output')
    # Create annotations
    text = clean_string(read_file(config['document']))
    use_characters = get_user_info("use_characters")
    if use_characters:
        from narrateit.dialogue import annotate_with_llm
        from narrateit.transformer_util import load_model_tokenizer
        from narrateit.peft_util import load_peft_model
        
        load_annotations = get_user_info("load_annotations")
        if load_annotations:
            with open(config['annotation_file'], 'r', encoding='utf-8') as f:
                annotations = json.load(f)
        else:
            model_path = config['annotation_model']
            model_, tokenizer = load_model_tokenizer(model_path)
            model = model_
            if 'peft_model' in config and config['peft_model']:
                model = load_peft_model(model_, config['peft_model'])
            annotations = annotate_with_llm(text, model, tokenizer, model_)
            del model
            del model_
            del tokenizer
            import torch
            torch.cuda.empty_cache()
            gc.collect()
    else:
        annotations = annotate_without_llm(text)
    return annotations


def narrate(annotations):
    from narrateit.tts.xtts_util import create_wave_files, load_tts
    from narrateit.narrate import merge_audio
    # Create wave files
    model, tokenizer = load_tts(config['tts_model'])
    create_wave_files(annotations, model, tokenizer)
    # Create final audio
    merged_audio = merge_audio()
    output_format = 'mp3'
    output_file = '.'.join(os.path.split(config['document'])[1].split('.')[:-1] + [output_format])
    merged_audio.export(os.path.join('output', output_file), format=output_format)


def main():
    parser = argparse.ArgumentParser(description='Process some data.')
    parser.add_argument('--annotate', action='store_true', help='Run the annotate function')
    parser.add_argument('--characters', action='store_true', help='Run the resolve_characters function')
    parser.add_argument('--narrate', action='store_true', help='Run the narrate function')

    args = parser.parse_args()
    
    # Only annotate
    if args.annotate:
        annotations = annotate()
        save_json(config['annotation_file'], annotations)
    # Only characterize
    if args.characters:
        with open(config['annotation_file'], 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        annotations = resolve_characters(annotations)
        save_json(config['annotation_file'], annotations)
    # Only narrate
    if args.narrate:
        with open(config['annotation_file'], 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        narrate(annotations)
    # Do the standard pipeline in case no keywords are given
    if not any(vars(args).values()):
        annotations = annotate()
        save_json(config['annotation_file'], annotations)
        annotations = resolve_characters(annotations)
        save_json(config['annotation_file'], annotations)
        narrate(annotations)


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
