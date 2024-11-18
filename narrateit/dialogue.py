import os
import re
import json
import warnings
from tqdm import tqdm

import torch
from transformers import StoppingCriteriaList

from .util import clean_string, config, clean_annotations, read_file
from .annotate import extract_characters
from .prompt_util import get_prompt
from .transformer_util import load_model_tokenizer, get_sentences


can_generate_state = True


class LLMOutputWarning(Warning):
    pass


def get_llm_state(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors='pt').to("cuda")
    state = None
    global can_generate_state
    try:
        output = model(**inputs)
        state = output.past_key_values
    except Exception as e:
        if can_generate_state:
            print("Warning: Could not generate state!")
            print(e)
        can_generate_state = False
    return state


def get_context(fragments, i, forward, min_context_length=200):
    """
    ''.join(fragments) is a literary text. This function returns the text before or after the i'th text fragment with a
    minimum length of min_context_length. forward specifies whether to find the context after the i'th fragment or
    before. Special cases are when i is so small or large that the context is shorter than the min_context_length
    or even empty. Only whole fragments are added to the context window
    """
    if forward:
        context_fragments = fragments[i+1:]
    else:
        context_fragments = fragments[:i][::-1]
    context = []
    current_length = 0
    for fragment in context_fragments:
        context.append(fragment)
        current_length += len(fragment)
        if current_length >= min_context_length:
            break
    if not forward:
        context.reverse()
    return ''.join(context)


def classify_dialogue(prompt, model, tokenizer, state=None):
    inputs = tokenizer(prompt, return_tensors='pt').to("cuda")
    if state is not None:
        inputs = {"input_ids": inputs["input_ids"],
                  "past_key_values": state}
    next_logits = model(**inputs)['logits'][:, -1:]
    top10_logits, top10 = torch.topk(next_logits[0, 0], 10)
    pvalues = torch.softmax(top10_logits, dim=-1)
    decoded = tokenizer.batch_decode(top10.reshape(-1, 1), skip_special_tokens=True)
    for t in decoded:
        char = t.strip().lower()[0]
        if char in ['d', 'n', 'b', 'c']:
            break
    return {'n': 0, 'd': 1, 'b': 2, 'c': 1}.get(char, 2), decoded, pvalues.detach().cpu().numpy()


# TODO Delete redundant classification method
def classify_document(fragments, model, tokenizer, character_model=None, i_min=46, i_max=48):
    if character_model is None: # TODO: Use character model to determine voice characteristics
        character_model = model
    config['eot_token'] = ''  # TODO: Create special token
    config['sot_token'] = ''  # TODO: Create special token
    replacements = {"DOCUMENT_TYPE": config['document_type']}
    dialogue_prompt = read_file(config['dialogue_prompt'])
    
    try:
        eos_token = tokenizer.eos_token  # eos_token = '<end_of_turn>'
    except AttributeError:
        from transformer_util import get_end_of_sequence_token
        eos_token = get_end_of_sequence_token(model, tokenizer)
    config['eos_token'] = eos_token
    
    # init loop
    classifications = []
    characters = {}
    last_characters = []
    last_genders = []
    # init prompt
    replacements["EOS"] = eos_token
    replacements["BRACKET_OPEN"] = '<<<'
    replacements["BRACKET_CLOSE"] = '>>>'
    system_prompt = get_prompt(dialogue_prompt, characters, last_characters, last_genders, replacements)
    llm_state = get_llm_state(system_prompt, model, tokenizer)
    samples_output = '.'.join(config['document'].split('.')[:-1])
    if not os.path.exists(samples_output):
        os.mkdir(samples_output)
    for i in tqdm(range(i_min, i_max)):
        text = fragments[i]
        pretext = get_context(fragments, i, forward=False)
        posttext = get_context(fragments, i, forward=True)
        prompt = system_prompt + 'Input:\n' + pretext + replacements["BRACKET_OPEN"] + text + replacements["BRACKET_CLOSE"] + posttext + '\n\nOutput: \n'
        result, tokens, pvalues = classify_dialogue(prompt, model, tokenizer, state=llm_state)
        token_ps = [f'{t_}: {p_*100:.1f}' for t_, p_ in zip(tokens, pvalues)]
        classifications.append(result)
        cl = {0: 'narration', 1: 'dialogue', 2: 'both'}[result]
        print(pretext + '<<<' + text + '>>>' + posttext + f'\n____________\n{token_ps}\n##############################')
        if i == 50:
            break
    return classifications


def classify_dialogue_with_explanation(prompt, model, tokenizer, state=None, allow_both=True):
    inputs = tokenizer(prompt, return_tensors='pt').to("cuda")
    
    def custom_stopping_criteria(input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs) -> bool:
        return "Output:\n" in tokenizer.decode(input_ids[0][-6:], skip_special_tokens=True)
    
    stopping_criteria = StoppingCriteriaList([custom_stopping_criteria])
    outputs = model.generate(**inputs, do_sample=False, temperature=None, top_p=None, max_new_tokens=200,
                             stopping_criteria=stopping_criteria, past_key_values=state,
                             return_dict_in_generate=True, pad_token_id=tokenizer.pad_token_id, output_scores=True)
    generated_text = tokenizer.decode(outputs.sequences[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    past_key_values = outputs.past_key_values
    if generated_text[-8:] == 'Output:\n':
        s = prompt + generated_text
    else:
        warnings.warn(f"LLM analysis didn't finish after maximum token length. Generated Analysis:\n{generated_text}", LLMOutputWarning)
        s = prompt + generated_text + '\nOutput:\n'
    explanation = generated_text[:-8]
    inputs = tokenizer(s, return_tensors='pt').to("cuda")
    inputs = {"input_ids": inputs["input_ids"], "past_key_values": past_key_values}
    next_logits = model(**inputs)['logits'][:, -1:]
    top10_logits, top10 = torch.topk(next_logits[0, 0], 10)
    pvalues = torch.softmax(top10_logits, dim=-1)
    decoded = tokenizer.batch_decode(top10.reshape(-1, 1), skip_special_tokens=True)
    for t in decoded:
        char = t.strip().lower()[0]
        if char in ['n', 'd'] or (char == 'b' and allow_both):
            break
    is_dialogue_or_both = {'n': 0, 'd': 1, 'b': 2}.get(char, 2)
    if is_dialogue_or_both == 1:
        s = s + 'Dialogue\n\nName:\n'
        inputs = tokenizer(s, return_tensors='pt').to("cuda")
        
        def custom_stopping_criteria(input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs) -> bool:
            return "\n" in tokenizer.decode(input_ids[0][-1:], skip_special_tokens=True).lower()
        
        stopping_criteria = StoppingCriteriaList([custom_stopping_criteria])
        # TODO: Use past_key_values state instead of the old state. Currently, not possible with generate() because
        #  past_key_values (coming from generate) is a Dynamic_Cache object while state (coming from forward) is a tuple
        outputs = model.generate(**inputs, do_sample=False, temperature=None, top_p=None,  max_new_tokens=20,
                                 past_key_values=state, stopping_criteria=stopping_criteria,
                                 return_dict_in_generate=True,
                                 pad_token_id=tokenizer.pad_token_id, output_scores=True)
        generated_text = tokenizer.decode(outputs.sequences[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        name = generated_text.split('\n')[0].strip()
    elif is_dialogue_or_both == 0:
        name = 'Narration'
    else:
        name = 'UNKNOWN'
        
    return is_dialogue_or_both, name, explanation, decoded, pvalues.detach().cpu().numpy()


def classify_document_with_explanation(fragments, model, tokenizer, character_model=None, i_min=0, i_max=None, verbose=0):
    if i_max is None:
        i_max = len(fragments)
    if character_model is None:  # TODO: Use character model to determine voice characteristics
        character_model = model
    config['eot_token'] = ''  # TODO: Create special token
    config['sot_token'] = ''  # TODO: Create special token
    replacements = {"DOCUMENT_TYPE": config['document_type']}
    dialogue_prompt = read_file(config['dialogue_explanation_prompt'])
    dialogue_prompt_both = read_file(config['dialogue_explanation_prompt_both'])
    
    try:
        eos_token = tokenizer.eos_token  # eos_token = '<end_of_turn>'
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
    replacements["BRACKET_OPEN"] = '<<<'
    replacements["BRACKET_CLOSE"] = '>>>'
    system_prompt = get_prompt(dialogue_prompt, characters, last_characters, last_genders, replacements)
    system_prompt_both = get_prompt(dialogue_prompt_both, characters, last_characters, last_genders, replacements)
    llm_state = get_llm_state(system_prompt, model, tokenizer)
    llm_state_both = get_llm_state(system_prompt_both, model, tokenizer)
    samples_output = '.'.join(config['document'].split('.')[:-1])
    if not os.path.exists(samples_output):
        os.mkdir(samples_output)
    for i in tqdm(range(i_min, i_max)):
        text = fragments[i]
        # Copy last result in case the string has no alphanumeric symbols
        if not any(c.isalnum() for c in text):
            continue
        text = text.replace('\n', ' ').replace('\r', ' ')
        pretext = get_context(fragments, i, forward=False)
        posttext = get_context(fragments, i, forward=True)
        prompt = system_prompt_both + 'Input:\n' + pretext + replacements["BRACKET_OPEN"] + text + replacements[
            "BRACKET_CLOSE"] + posttext + '\n\nAnalysis:\n'
        result, name, explanation, tokens, pvalues = classify_dialogue_with_explanation(prompt, model, tokenizer,
                                                                                        state=llm_state_both,
                                                                                        allow_both=True)
        if verbose:
            cl = {0: 'Narration', 1: 'Dialogue', 2: 'Both'}[result]
            token_ps = [f'{t_}: {p_ * 100:.1f}' for t_, p_ in zip(tokens, pvalues)]
            print('Input:\n' + pretext + '<<<' + text + '>>>' + posttext + '\n\nAnalysis:\n' + explanation)
            print('\nOutput:\n' + cl)
            if result == 1:
                print('\nName:\n' + name)
            print(f'\n____________\n{token_ps}\n##############################')
        if result < 2:
            annotations.append({'speaker': name, 'text': text})
        else:
            # Solve the ambiguous or non classified fragment word by word
            splits = text.split(' ')
            words = []
            word = ''
            for j, split in enumerate(splits):
                word += split + ' '
                # Check for any letter or number (things that can be spoken)
                if any(c.isalnum() for c in word):
                    words.append(word)
                    word = ''
                elif j == len(splits):
                    words[-1] += word + ' '
            words[-1] = words[-1][:-1]
            for j in range(len(words)):
                pretext_ = pretext + ''.join(words[:j])
                posttext_ = ''.join(words[j+1:]) + posttext
                prompt = system_prompt + 'Input:\n' + pretext_ + replacements["BRACKET_OPEN"] + words[j] + replacements[
                    "BRACKET_CLOSE"] + posttext_ + '\n\nExplanation:\n'
                result, name, explanation, tokens, pvalues = classify_dialogue_with_explanation(prompt, model,
                                                                                                tokenizer,
                                                                                                state=llm_state,
                                                                                                allow_both=False)
                if verbose:
                    cl = {0: 'Narration', 1: 'Dialogue', 2: 'Not Classified'}[result]
                    token_ps = [f'{t_}: {p_ * 100:.1f}' for t_, p_ in zip(tokens, pvalues)]
                    print("Resolving 'Both' conflict_________________________")
                    print('Input:\n' + pretext_ + '<<<' + words[j] + '>>>' + posttext_ + '\n\nExplanation:\n' + explanation)
                    print('\nOutput:\n' + cl)
                    if result == 1:
                        print('\nName:\n' + name)
                    print(f'\n____________\n{token_ps}\n##############################')
                if result == 2:
                    # Classify unsuccessfully processed words as Narration
                    name = "Narration"
                annotations.append({'speaker': name, 'text': words[j]})
    return clean_annotations(annotations)


def get_fragments(sentences):
    """document is a complete text. sentences are split it into chunks we would recognize as sentences using a
    nontrivial way to split not just based on ".!?". After that, we further split into fragments by splitting along
    dialogue indicating symbols such as ["“”]. In this way, the sentence:
     Suddenly Thomas cried "I can't deal with this anymore!" with a booming voice.
    becomes three segments:
    ['Suddenly Thomas cried ', '"I can't deal with this anymore!"', ' with a booming voice.']
    No symbols are removed. If the fragments were to be joined via ''.join(fragments) it is the entire text
    just as ''.join(sentences) also is.
    """
    fragments = []
    for sentence in sentences:
        parts = re.split(r'(["“”])', sentence)
        fragment = ""
        quote_open = True
        if len(parts) > 0 and len(parts) % 2 == 0:
            parts[-2] = ''.join(parts[-2:])
            parts = parts[:-1]
        for part in parts:
            if part in ['"', '“', '”']:
                if quote_open:
                    if fragment:
                        fragments.append(fragment)
                    fragment = part
                else:
                    fragment += part
                    fragments.append(fragment)
                    fragment = ""
                quote_open = not quote_open
            else:
                fragment += part
        if fragment:
            fragments.append(fragment)
    return fragments


def annotate_with_llm(text, model, tokenizer, character_model=None):
    if character_model is None:
        character_model = model
    sentences = get_sentences(text)
    fragments = get_fragments(sentences)
    annotations = classify_document_with_explanation(fragments, model, tokenizer)
    characters = extract_characters(annotations)
    # character_prompt = read_file(config['character_prompt'])
    # characters = generate_genders(character_prompt, characters, model, tokenizer)
    result = {'annotation': annotations, 'characters': characters, 'characters_resolved': False}
    return result


if __name__ == '__main__':
    document_ = clean_string(read_file(config['document']))
    sentences_ = get_sentences(document_)
    fragments_ = get_fragments(sentences_)
    model_path = config['annotation_model']
    model_, tokenizer_ = load_model_tokenizer(model_path)
    config = json.load(open(os.path.join('config', 'config.json')))
    annotations_ = classify_document_with_explanation(fragments_, model_, tokenizer_, i_min=20, i_max=40, verbose=1)
