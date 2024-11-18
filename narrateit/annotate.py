import os
import re
import codecs
import json
import time
# from json_extractor import JsonExtractor
from .util import write_to_file, clean_annotations, save_json, clean_string, config, read_file
from .prompt_util import get_prompt, append_annotations, create_dual_segments, append_previous_annotations

from tqdm import tqdm
from .transformer_util import load_model_tokenizer, generate_genders, init_transformer_states, get_sentences
# from .transformer_util import generate_annotations as generate_text
# from .transformer_util import generate_based_on_state as generate_text
from .transformer_util import generate_json


class BadLLMOutput(Exception):
    pass


def extract_characters(annotations):
    characters = {}
    for a in annotations:
        name = a['speaker']
        if name not in characters and name.lower() != 'narration':
            characters[name] = {'name': name, 'n_dialogue': 1}
        elif name.lower() != 'narration':
            characters[name]['n_dialogue'] += 1
    return characters


def get_annotation(old_text, new_text, prompt, old_annotations, model, tokenizer):
    def extract_from_text(s):
        s = re.sub(r'(\r?\n)+', '', s)
        try:
            return json.loads(s)["annotation"]
        except json.decoder.JSONDecodeError:  # TODO: Save program state (somehow)
            print("Error in LLM output format, illegal json. Aborting...")
            raise BadLLMOutput(f"Bad LLM output: \n######{s}\n######")
    
    previous_output = append_previous_annotations(old_annotations)
    gen_json, _ = generate_json(prompt, old_text + new_text, previous_output, model, tokenizer, len(new_text) + 200)
    in_and_out = prompt + "input: " + old_text + new_text + "\noutput: " + previous_output + gen_json[22:]
    write_to_file(f"inputs_plus_outputs_tmp/prompt_plus_generated_{int(time.time())}", in_and_out)  # TODO: DELETE THIS
    annotations = extract_from_text(gen_json)
    return annotations


def annotate_with_llm(segments, model, tokenizer, character_model=None):
    if character_model is None:
        character_model = model
    config['eot_token'] = ''  # TODO: Create special token
    config['sot_token'] = ''  # TODO: Create special token
    replacements = {"DOCUMENT_TYPE": config['document_type']}
    system_prompt = read_file(config['system_prompt'])
    
    try:
        eos_token = tokenizer.eos_token
        # eos_token = '<end_of_turn>'
    except AttributeError:
        from transformer_util import get_end_of_sequence_token
        eos_token = get_end_of_sequence_token(model, tokenizer)
    config['eos_token'] = eos_token
    
    # init loop
    annotations = []
    characters = {}
    last_characters = []
    last_genders = []
    previous_text = ""
    previous_annotations = []
    # init prompt
    replacements["EOS"] = eos_token
    prompt = get_prompt(system_prompt, characters, last_characters, last_genders, replacements)
    # init_transformer_states(prompt[:-1], model, tokenizer)
    samples_output = '.'.join(config['document'].split('.')[:-1])
    if not os.path.exists(samples_output):
        os.mkdir(samples_output)
    for i in tqdm(range(len(segments))):
        text = segments[i]
        try:
            result = get_annotation(previous_text, text, prompt, previous_annotations, model, tokenizer)
        except BadLLMOutput as e:  # TODO: Handle this
            print(e)
            continue
        result = clean_annotations(result)
        previous_annotations = result
        annotations += result
        previous_text = text
        
        save_json(os.path.join(samples_output, f"segment_{i}.json"), {'text': text, 'annotation': result})
        
        # annotations += result[1]
        # Find last n speakers
        # last_characters = []
        # last_genders = []
        # for x in annotations[::-1]:
        #     if len(last_characters) == config['number_of_last_character_in_prompt']:
        #         break
        #     character = x['speaker']
        #     if character not in last_characters and character != "NARRATION":
        #         last_characters.append(character)
        #         last_genders.append('m')
        #     if character not in characters and character != "NARRATION":
        #         characters[character] = {}
        # print("$$$$$$$")
        # print(last_characters)
        # print(last_genders)
        # print("$$$$$$$")
        if i == 50:
            break
    annotations = clean_annotations(annotations)
    characters = extract_characters(annotations)
    character_prompt = read_file(config['character_prompt'])
    characters = generate_genders(character_prompt, characters, model, tokenizer)
    result = {'annotation': annotations, 'characters': characters, 'characters_resolved': False}
    return result


def test_llm(model, tokenizer, examples):
    def extract_from_text(s):
        s = re.sub(r'(\r?\n)+', '', s)
        try:
            return json.loads(s)["annotation"]
        except json.decoder.JSONDecodeError:
            raise BadLLMOutput(f"Bad LLM output: \n######{s}\n######")
    
    replacements = {"DOCUMENT_TYPE": config['document_type']}
    system_prompt = read_file(config['system_prompt'])
    print(f"Testing prompt: {config['system_prompt']}")
    texts = [e['text'] for e in examples]
    
    try:
        eos_token = tokenizer.eos_token
        # eos_token = '<end_of_turn>'
    except AttributeError:
        from transformer_util import get_end_of_sequence_token
        eos_token = get_end_of_sequence_token(model, tokenizer)
    config['eos_token'] = eos_token
    
    # init loop
    annotations = []
    characters = {}
    last_characters = []
    last_genders = []
    # init prompt
    replacements["EOS"] = eos_token
    prompt = get_prompt(system_prompt, characters, last_characters, last_genders, replacements)
    # init_transformer_states(prompt[:-1], model, tokenizer)
    for i, s in tqdm(enumerate(texts)):
        prompt_app = config['eos_token'] + "\n\n"
        previous_output = "{\n    \"annotation\": [\n"
        gen_json, _ = generate_json(prompt + prompt_app, s, previous_output, model, tokenizer, len(s) + 200)
        in_and_out = prompt + prompt_app + "input: " + s + "\noutput: " + gen_json
        write_to_file(f"inputs_plus_outputs_tmp/test_examples_{int(time.time())}", in_and_out)
        try:
            annotation = extract_from_text(gen_json)
        except BadLLMOutput as e:
            print(e)
            annotations.append([])
            continue
        annotation = clean_annotations(annotation)
        annotations.append(annotation)
        save_json(f"test_llm_output/samples_output_{i}.json", {'text': s, 'annotation': annotation})
    return annotations


def compare_annotations(truth, generated):
    """
    truth: list with dicts, [{'text': s, 'annotation': [...]}, {}, ...]
    generated: list with lists [[...], [...], ...]
    Here, [...] are annotations. Entries are dicts {"speaker": "name", "text": "spoken text"}
    """
    assert(len(truth) == len(generated))
    n_total = len(truth)
    success = []
    words_correct = []
    for i in range(len(truth)):
        # text = truth[i]['text']
        annotation_true = truth[i]['annotation']
        annotation_test = generated[i]
        sid = 0
        is_correct = len(annotation_test) == len(annotation_true)
        words_correct_ = is_correct
        while is_correct and sid < len(annotation_test):
            is_correct = annotation_true[sid]['speaker'].lower() == annotation_test[sid]['speaker'].lower()
            words_correct_ = is_correct and words_correct_
            if words_correct_:
                # check the text:
                words_true = re.sub(r'[,\s]', '', annotation_true[sid]['text']).lower()
                words_test = re.sub(r'[,\s]', '', annotation_test[sid]['text']).lower()
                words_correct_ = words_true == words_test
                if not words_correct_:
                    print(f"Words unaligned in sample {i}:\ns1={words_true}\ns2={words_test}")
            sid += 1
        success.append(is_correct)
        words_correct.append(words_correct_)
    return success, words_correct


def test_prompt(model, tokenizer):
    # test with examples
    examples = json.load(open(os.path.join('prompts', 'test_examples.json')))
    annotations = test_llm(model, tokenizer, examples)
    success, words_correct = compare_annotations(examples, annotations)
    n_correct = sum(success)
    n_words_correct = sum(words_correct)
    n_total = len(annotations)
    with open('.'.join(config['system_prompt'].split('.')[:-1]) + '_results.txt', 'w', encoding='utf-8') as file:
        file.writelines([f'{n_correct}/{len(annotations)}\n', f'{n_words_correct}/{n_correct}'])
    print(f"speaker annotations correct {n_correct}/{n_total}")
    print(f"spoken words that are correct {n_words_correct}/{n_correct}")
    print([(i, s, w) for i, s, w in zip(range(len(success)), success, words_correct)])


def helper_convert_list_to_annotations(file, output_file):
    """Input is json with list at top level. Each entry is an annotated segment.
    It has text and annotation. An annotation is a list with entries object. They have speaker and text"""
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    annotations = []
    for x in data:
        annotations += x['annotation']
    annotations = clean_annotations(annotations)
    characters = extract_characters(annotations)
    # character_prompt = read_file(config['character_prompt'])
    # characters = generate_genders(character_prompt, characters, model, tokenizer)
    result = {'annotation': annotations, 'characters': characters}
    save_json(output_file, result)


def unload_llm():
    global model
    global tokenizer
    del model
    del tokenizer
    import torch
    torch.cuda.empty_cache()


if __name__ == "__main__":
    document = clean_string(read_file(config['document']))
    # sentences = re.split(r'(?<=[.!?])', document)
    sentences = get_sentences(document)
    segments = create_dual_segments(sentences, config['min_segment_length'], config['min_overlap_length'])
    
    model_path = config['annotation_model']
    model, tokenizer = load_model_tokenizer(model_path)
    # test_prompt(model, tokenizer)
    
    # annotate a text:
    result = annotate_with_llm(segments, model, tokenizer)
    
    # load a peft model
    # from peft_util import load_peft_model
    # peft_model = load_peft_model(model)
