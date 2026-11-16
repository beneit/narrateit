import os
import time
from tqdm import tqdm
import json
import warnings

import torch
from transformers import StoppingCriteriaList

from ..util import config, read_file, clean_annotations, write_to_file
from ..annotate import BaseModel
from ..transformer_util import load_model_tokenizer, get_sentences
from ..prompt_util import get_fragments, get_prompt, create_structured_fewshot, get_context


class LLMOutputWarning(Warning):
    pass


class LLMStateWarning(Warning):
    pass


class TransformerWithStructuredExplanations(BaseModel):
    can_generate_state = True
    structure = [['prefix', "<Text before section:> "], ['section', "<Text to analyse:> "],
                 ['suffix', "<Text after section:> "], ['analysis', "<Reasoning:> "], ['output', "<Output:> "],
                 ['name', "<Name:> "]]
    
    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        if model is None or tokenizer is None:
            self.model, self.tokenizer = load_model_tokenizer(config['annotation_model'])
        if 'character_model' in config:
            self.character_model, self.character_tokenizer = load_model_tokenizer(config['character_model'])
        else:
            self.character_model, self.character_tokenizer = self.model, self.tokenizer
    
    def get_llm_state(self, prompt):
        if not self.can_generate_state:
            return None
        inputs = self.tokenizer(prompt, return_tensors='pt').to("cuda")
        state = None
        try:
            output = self.model(**inputs)
            state = output.past_key_values
        except Exception as e:
            if self.can_generate_state:
                warnings.warn(f"Could not generate state:\n{e}", LLMStateWarning)
            self.can_generate_state = False
        # Test if model can use generated state with generate call
        if self.can_generate_state:
            try:
                _ = self.model.generate(**inputs, max_new_tokens=1, past_key_values=state)
            except Exception as e:
                self.can_generate_state = False
                warnings.warn(f"Could not use state in model.generate call:\n{e}", LLMStateWarning)
                state = None
        return state
    
    def print_call(self, result, tokens, pvalues):
        token_ps = [f'{t_}: {p_ * 100:.1f}' for t_, p_ in zip(tokens, pvalues)]
        for field, placement in self.structure:
            print(placement + result[field])
        print(f'\n____________\n{token_ps}\n##############################')
    
    def classify_dialogue(self, prompt, result, state=None, allow_both=True):
        # Step 1: Explanation
        s = prompt
        for i in range(3):
            s += self.structure[i][1] + result[self.structure[i][0]] + "\n"
        s += self.structure[3][1]
        
        inputs = self.tokenizer(s, return_tensors='pt').to("cuda")
        
        def custom_stopping_criteria(input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs) -> bool:
            match = self.structure[4][1].strip()
            return match in self.tokenizer.decode(input_ids[0][-len(match):], skip_special_tokens=True)
        
        stopping_criteria = StoppingCriteriaList([custom_stopping_criteria])
        outputs = self.model.generate(**inputs,
                                      do_sample=False,
                                      temperature=None,
                                      top_p=None,
                                      max_new_tokens=200,
                                      stopping_criteria=stopping_criteria,
                                      past_key_values=state,
                                      return_dict_in_generate=True,
                                      pad_token_id=self.tokenizer.pad_token_id,
                                      output_scores=True)
        generated_text = self.tokenizer.decode(outputs.sequences[0][len(inputs["input_ids"][0]):],
                                               skip_special_tokens=True)
        past_key_values = outputs.past_key_values
        if generated_text[-len(self.structure[4][1]):] == self.structure[4][1]:
            s += generated_text
        elif generated_text[-len(self.structure[4][1].strip()):] == self.structure[4][1].strip():
            s += generated_text[:-len(self.structure[4][1].strip())] + self.structure[4][1]
        else:
            warnings.warn(
                f"LLM analysis finished with maximum token length. Generated Analysis:\n{generated_text}",
                LLMOutputWarning)
            s += generated_text + '\n' + self.structure[4][1]
        result[self.structure[3][0]] = generated_text[:-len(self.structure[4][1])]
        
        # Step 2: Classification
        write_to_file(f"inputs_tmp/prompt_{int(time.time())}", s)  # TODO: Delete
        inputs = self.tokenizer(s, return_tensors='pt').to("cuda")
        inputs = {"input_ids": inputs["input_ids"], "past_key_values": past_key_values}
        next_logits = self.model(**inputs)['logits'][:, -1:]
        top10_logits, top10 = torch.topk(next_logits[0, 0], 10)
        pvalues = torch.softmax(top10_logits, dim=-1)
        decoded = self.tokenizer.batch_decode(top10.reshape(-1, 1), skip_special_tokens=True)
        for t in decoded:
            char = t.strip().lower()[0]
            if char in ['n', 'd'] or (char == 'b' and allow_both):
                break
        is_dialogue_or_both = {'n': 0, 'd': 1, 'b': 2}.get(char, 2)
        result[self.structure[4][0]] = {0: "Narration", 1: "Dialogue", 2: "Both"}[is_dialogue_or_both]
        
        # Step 3: Name
        if is_dialogue_or_both == 1:
            s += 'Dialogue\n' + self.structure[5][1]
            inputs = self.tokenizer(s, return_tensors='pt').to("cuda")
            
            def custom_stopping_criteria(input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs) -> bool:
                return "\n" in self.tokenizer.decode(input_ids[0][-1:], skip_special_tokens=True).lower()
            
            stopping_criteria = StoppingCriteriaList([custom_stopping_criteria])
            # TODO: Use past_key_values state instead of the old state. Currently, not possible with generate() because
            #  past_key_values (coming from generate) is a Dynamic_Cache object while state (coming from forward) is a
            #  tuple
            outputs = self.model.generate(**inputs,
                                          do_sample=False,
                                          temperature=None,
                                          top_p=None,
                                          max_new_tokens=20,
                                          past_key_values=state,
                                          stopping_criteria=stopping_criteria,
                                          return_dict_in_generate=True,
                                          pad_token_id=self.tokenizer.pad_token_id,
                                          output_scores=True)
            generated_text = self.tokenizer.decode(outputs.sequences[0][len(inputs["input_ids"][0]):],
                                                   skip_special_tokens=True)
            name = generated_text.split('\n')[0].strip()
        elif is_dialogue_or_both == 0:
            name = 'Narration'
        else:
            name = 'UNKNOWN'
        result[self.structure[5][0]] = name
        
        return decoded, pvalues.detach().cpu().to(torch.float32).numpy()
    
    def annotate(self, text, i_min=0, i_max=None, verbose=0):
        sentences = get_sentences(text)
        fragments = get_fragments(sentences)
        # fragments = list(map(lambda s: s.replace('\n', ' ').replace('\r', ' '), fragments))
        if i_max is None:
            i_max = len(fragments)
        
        replacements = {"DOCUMENT_TYPE": config['document_type']}
        
        examples = json.load(open(config["examples"]))
        examples = ''.join(create_structured_fewshot(examples, self.structure))
        dialogue_prompt = read_file(config[config["prompt_strategy"]]["system_prompt_single"]) + examples
        dialogue_prompt_both = read_file(config[config["prompt_strategy"]]["system_prompt_both"]) + examples
        
        try:
            eos_token = self.tokenizer.eos_token  # eos_token = '<end_of_turn>'
        except AttributeError:
            from ..transformer_util import get_end_of_sequence_token
            eos_token = get_end_of_sequence_token(self.model, self.tokenizer)
        
        # init loop
        annotations = []
        characters = {}
        last_characters = []
        last_genders = []
        # init prompt
        replacements["EOS"] = eos_token
        system_prompt_single = get_prompt(dialogue_prompt, characters, last_characters, last_genders, replacements)
        system_prompt_both = get_prompt(dialogue_prompt_both, characters, last_characters, last_genders, replacements)
        llm_state_single = self.get_llm_state(system_prompt_single)
        llm_state_both = self.get_llm_state(system_prompt_both)
        
        def clean_breaks(s):
            return s.replace('\n', ' ').replace('\r', ' ')
        
        for i in tqdm(range(i_min, i_max)):
            fragment = fragments[i]
            # Copy last result in case the string has no alphanumeric symbols
            if not any(c.isalnum() for c in fragment):
                continue
            prefix = get_context(fragments, i, forward=False)
            suffix = get_context(fragments, i, forward=True)
            section = {self.structure[0][0]: prefix, self.structure[1][0]: fragment, self.structure[2][0]: suffix}
            tokens, pvalues = self.classify_dialogue(system_prompt_both, section, state=llm_state_both, allow_both=True)
            
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
                    tokens, pvalues = self.classify_dialogue(system_prompt_single, section, state=llm_state_single,
                                                             allow_both=False)
                    name = section[self.structure[5][0]]
                    if verbose:
                        self.print_call(section, tokens, pvalues)
                    if name == "UNKNOWN":
                        # Classify unsuccessfully processed words as Narration
                        name = "Narration"
                    annotations.append({'speaker': name, 'text': words[j]})
        annotations = clean_annotations(annotations)
        return annotations
