import os
import time
from tqdm import tqdm
import json
import warnings

import numpy as np
from openai import OpenAI

from ..util import config, read_file, clean_annotations, write_to_file
from ..annotate import BaseModel
from ..prompt_util import get_fragments, get_prompt, create_structured_fewshot, get_context, get_sentences


class LLMOutputWarning(Warning):
    pass


class OpenAIWithStructuredExplanations(BaseModel):
    can_generate_state = False
    structure = [['prefix', "<Text before section:> "], ['section', "<Text to analyse:> "],
                 ['suffix', "<Text after section:> "], ['analysis', "<Reasoning:> "], ['output', "<Output:> "],
                 ['name', "<Name:> "]]
    
    def __init__(self, model="gpt-3.5-turbo-instruct"):
        self.client = OpenAI()
        self.model = model
    
    def print_call(self, result, tokens, pvalues):
        token_ps = [f'{t_}: {p_ * 100:.1f}' for t_, p_ in zip(tokens, pvalues)]
        for field, placement in self.structure:
            print(placement + result[field])
        print(f'\n____________\n{token_ps}\n##############################')
    
    def classify_dialogue(self, prompt, result, allow_both=True):
        # Step 1: Explanation
        s = prompt
        for i in range(3):
            s += self.structure[i][1] + result[self.structure[i][0]] + "\n"
        s += self.structure[3][1]
        
        stopping_criteria = [self.structure[4][1].strip()]
        completion = self.client.completions.create(model=self.model,
                                                    prompt=s,
                                                    max_tokens=200,
                                                    temperature=0,
                                                    stop=stopping_criteria)
        generated_text = completion["choices"][0]["text"]
        s += generated_text + '\n' + self.structure[4][1]
        if completion["choices"][0]["finish_reason"] == "length":
            warnings.warn(
                f"LLM analysis didn't finished with maximum token length. Generated Analysis:\n{generated_text}",
                LLMOutputWarning)
        result[self.structure[3][0]] = generated_text
        
        # Step 2: Classification
        write_to_file(f"inputs_tmp/prompt_{int(time.time())}", s)  # TODO: Delete
        completion = self.client.completions.create(model=self.model,
                                                    prompt=s,
                                                    max_tokens=1,
                                                    logprobs=10)
        topn_tokens_logits = completion["choices"][0]["logprobs"]["top_logprobs"][0]
        topn_tokens = list(topn_tokens_logits.keys())
        topn_logits = np.array([topn_tokens_logits[token] for token in topn_tokens])

        def softmax(logits):
            exp_logits = np.exp(logits - np.max(logits))  # Subtract max logit for numerical stability
            return exp_logits / exp_logits.sum
        
        topn_probs = softmax(topn_logits)
        
        for t in topn_tokens:
            char = t.strip().lower()[0]
            if char in ['n', 'd'] or (char == 'b' and allow_both):
                break
        is_dialogue_or_both = {'n': 0, 'd': 1, 'b': 2}.get(char, 2)
        result[self.structure[4][0]] = {0: "Narration", 1: "Dialogue", 2: "Both"}[is_dialogue_or_both]
        
        # Step 3: Name
        if is_dialogue_or_both == 1:
            s += 'Dialogue\n' + self.structure[5][1]
            stopping_criteria = ["\n"]
            completion = self.client.completions.create(model=self.model,
                                                        prompt=s,
                                                        max_tokens=20,
                                                        temperature=0,
                                                        stop=stopping_criteria)
            generated_text = completion["choices"][0]["text"]
            name = generated_text.strip()
        elif is_dialogue_or_both == 0:
            name = 'Narration'
        else:
            name = 'UNKNOWN'
        result[self.structure[5][0]] = name

        return topn_tokens, topn_probs
    
    def annotate(self, text, i_min=0, i_max=None, verbose=0):
        sentences = get_sentences(text)
        fragments = get_fragments(sentences)
        if i_max is None:
            i_max = len(fragments)

        replacements = {"DOCUMENT_TYPE": config['document_type']}

        examples = json.load(open(config["examples"]))
        examples = ''.join(create_structured_fewshot(examples, self.structure))
        dialogue_prompt = read_file(config[config["prompt_strategy"]]["system_prompt_single"]) + examples
        dialogue_prompt_both = read_file(config[config["prompt_strategy"]]["system_prompt_both"]) + examples

        eos_token = "<|endoftext|>"
        replacements["EOS"] = eos_token
        
        # init loop
        annotations = []
        characters = {}
        last_characters = []
        last_genders = []
        # init prompt
        system_prompt_single = get_prompt(dialogue_prompt, characters, last_characters, last_genders, replacements)
        system_prompt_both = get_prompt(dialogue_prompt_both, characters, last_characters, last_genders, replacements)
        
        for i in tqdm(range(i_min, i_max)):
            fragment = fragments[i]
            # Copy last result in case the string has no alphanumeric symbols
            if not any(c.isalnum() for c in fragment):
                continue
            fragment = fragment.replace('\n', ' ').replace('\r', ' ')
            prefix = get_context(fragments, i, forward=False)
            suffix = get_context(fragments, i, forward=True)
            section = {self.structure[0][0]: prefix, self.structure[1][0]: fragment, self.structure[2][0]: suffix}
            tokens, pvalues = self.classify_dialogue(system_prompt_both, section, allow_both=True)
            
            if verbose:
                self.print_call(section, tokens, pvalues)
            if section[self.structure[4][0]] in ['Dialogue', 'Narration']:
                annotations.append({'speaker': section[self.structure[5][0]], 'text': fragment})
            else:
                # Solve the ambiguous or non classified fragment word by word
                splits = fragment.split(' ')
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
                    prefix_ = prefix + ''.join(words[:j])
                    suffix_ = ''.join(words[j + 1:]) + suffix
                    section = {self.structure[0][0]: prefix_, self.structure[1][0]: words[j],
                               self.structure[2][0]: suffix_}
                    tokens, pvalues = self.classify_dialogue(system_prompt_single, section, allow_both=False)
                    name = section[self.structure[5][0]]
                    if verbose:
                        self.print_call(section, tokens, pvalues)
                    if name == "UNKNOWN":
                        # Classify unsuccessfully processed words as Narration
                        name = "Narration"
                    annotations.append({'speaker': name, 'text': words[j]})
        annotations = clean_annotations(annotations)
        return annotations
  