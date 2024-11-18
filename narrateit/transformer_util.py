import os
import json
import re
import time
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList

from .util import write_to_file

torch.random.manual_seed(0)


def load_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.appended_token = int(tokenizer('}', return_tensors='pt').to('cuda')['input_ids'][0][-1])
    tokenizer.curly1 = int(tokenizer('{', return_tensors='pt').to('cuda')['input_ids'][0][-1])
    tokenizer.curly2 = int(tokenizer('}', return_tensors='pt').to('cuda')['input_ids'][0][-1])
    return tokenizer


def load_model_tokenizer(model_path):
    print(f"Loading model '{model_path}'...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = load_tokenizer(model_path)
    model.past_states__ = None
    return model, tokenizer


def init_transformer_states(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors='pt').to("cuda")
    # inputs = model.prepare_inputs_for_generation(inputs)
    # output = model(**inputs, return_dict=True)
    output = model.generate(**inputs, do_sample=False, max_new_tokens=1, return_dict_in_generate=True,
                            # cache_implementation=None
                            )
    print(f"Initializing state: generated token: {int(output['sequences'][0][-1])}; expected token: {tokenizer.appended_token}")
    assert(int(output['sequences'][0][-1]) == tokenizer.appended_token)
    model.past_states__ = output.past_key_values


def generate_annotations(prompt, model, tokenizer, max_new_tokens):
    max_length = len(prompt)
    write_to_file(f"inputs_tmp/prompt_{int(time.time())}", prompt)  # TODO: Delete
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    count1 = prompt.count('{')
    count2 = prompt.count('}')
    
    def custom_stopping_criteria(input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs) -> bool:
        nonlocal count1, count2, tokenizer
        s = tokenizer.decode(input_ids[0][-1])
        count1 += s.count('{')
        count2 += s.count('}')
        # count1 = list(input_ids[0]).count(tokenizer.curly1)
        # count2 = list(input_ids[0]).count(tokenizer.curly2)
        # print("count({)=" + str(count1) + "; count(})=" + str(count2) + "-> " + str(count1 == count2))
        return count1 == count2
    
    stopping_criteria = StoppingCriteriaList([custom_stopping_criteria])
    
    outputs = model.generate(**inputs,
                             do_sample=False,
                             max_new_tokens=max_new_tokens,
                             stopping_criteria=stopping_criteria,
                             pad_token_id=tokenizer.pad_token_id)
    generated_text = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    return generated_text


def generate_based_on_state(prompt, model, tokenizer, max_new_tokens):
    write_to_file(f"inputs_tmp/prompt_{int(time.time())}", prompt)  # TODO: Delete
    inputs = tokenizer(prompt, return_tensors='pt').to("cuda")
    count1 = prompt.count('{')
    count2 = prompt.count('}')
    if model.past_states__ is None:
        raise ValueError("Model past_key_values not initialized.")
    
    def custom_stopping_criteria(input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs) -> bool:
        nonlocal count1, count2, tokenizer
        s = tokenizer.decode(input_ids[0][-1])
        count1 += s.count('{')
        count2 += s.count('}')
        return count1 == count2
    stopping_criteria = StoppingCriteriaList([custom_stopping_criteria])
    
    outputs = model.generate(**inputs,
                             do_sample=False,
                             past_key_values=model.past_states__,
                             max_new_tokens=max_new_tokens,
                             return_dict_in_generate=True,
                             # cache_implementation=None,
                             stopping_criteria=stopping_criteria)  # TODO: max_length, use_cache=True
    generated_text = tokenizer.decode(outputs.sequences[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    return generated_text


def generate_based_on_no_state(prompt, model, tokenizer, max_new_tokens=200):
    """We argmax the logits to get the next token. This neither generates nor utilizes past states (which is really bad
    Unfortunately, Gemma2 does not return states in the model call, nor can it read in states in the 'generate' call"""
    inputs = tokenizer(prompt, return_tensors='pt').to("cuda").input_ids
    original_length = len(inputs[0])
    count1 = prompt.count('{')
    count2 = prompt.count('}')
    for i in tqdm(range(max_new_tokens)):
        outputs = model(inputs, return_dict=True, use_cache=False)
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        s = tokenizer.decode(next_token_id)
        count1 += s.count('{')
        count2 += s.count('}')
        inputs = torch.cat([inputs, next_token_id.unsqueeze(-1)], dim=-1)
        if count1 == count2:
            break
    generated_text = tokenizer.decode(inputs[0, original_length:], skip_special_tokens=True)
    return generated_text


def generate_json(prompt, input_text, previous_output, model, tokenizer, max_new_tokens=200, past_key_values=None):
    """"""
    # Initialize the prompt
    prompt += f"input: {input_text}\noutput: {previous_output}"
    new_output = '{\n    "annotation": [\n'
    
    # Function to generate until a specific character is found
    def generate_until_char(model, tokenizer, prompt, stop_char, past_key_values=None):
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        def custom_stopping_criteria(input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs) -> bool:
            return stop_char in tokenizer.decode(input_ids[0][-1], skip_special_tokens=True)
        stopping_criteria = StoppingCriteriaList([custom_stopping_criteria])
        outputs = model.generate(**inputs, do_sample=False,
                                 max_new_tokens=max_new_tokens,
                                 stopping_criteria=stopping_criteria,
                                 past_key_values=past_key_values,
                                 return_dict_in_generate=True,
                                 pad_token_id=tokenizer.pad_token_id,
                                 output_scores=True)
        generated_text = tokenizer.decode(outputs.sequences[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        past_key_values = outputs.past_key_values
        return generated_text, past_key_values
    
    def check_next_token_is_comma(model, tokenizer, prompt, past_key_values=None):
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        # outputs = model(**inputs, past_key_values=past_key_values)
        # logits = outputs.logits[:, -1, :]
        # next_token_id = torch.argmax(logits, dim=-1).item()
        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=1,
                                 past_key_values=past_key_values, pad_token_id=tokenizer.pad_token_id,
                                 return_dict_in_generate=True, output_scores=True)
        # xx = tokenizer.decode(outputs.sequences[0])
        # print(xx)
        next_token_str = tokenizer.decode(outputs.sequences[0][-1], skip_special_tokens=True)
        return (len(next_token_str) > 0 and next_token_str[0] == ','), tokenizer.decode(outputs.sequences[0][-1], skip_special_tokens=False)
    
    ntstr = None  # TODO: Remove the ntstr variable
    while True:
        # Step 1: Write the initial part of the JSON element
        new_output += '        {\n            "speaker": "'
        prompt += '        {\n            "speaker": "'
        try:
            # Step 2: Generate until the next quote for speaker
            generated_text, past_key_values = generate_until_char(model, tokenizer, prompt, '"', past_key_values)
            speaker = generated_text.split('"')[0]
            new_output += speaker + '",\n            "text": "'
            prompt += speaker + '",\n            "text": "'
            
            # Step 3: Generate until the next quote for speech
            generated_text, past_key_values = generate_until_char(model, tokenizer, prompt, '"', past_key_values)
            text = generated_text.split('"')[0]
            new_output += text + '"\n        }'
            prompt += text + '"\n        }'
            
            # Step 4: Generate to check if there's a comma or end of list
            is_comma, ntstr = check_next_token_is_comma(model, tokenizer, prompt, past_key_values)
        except RuntimeError as e:
            print(f"Runtime error in generative process. PROMPT:\n{prompt}")
            print(f"\n_____________________________\nLAST_NEXT_TOKEN_STR_COMMA={ntstr}")
            is_comma = False
        if is_comma:
            new_output += ',\n'
            prompt += ',\n'
        else:
            new_output += '\n    ]\n}'
            break
    
    return new_output, past_key_values


def generate_genders(prompt, characters, model, tokenizer):
    for idc in characters:
        c = characters[idc]
        s = re.sub(r'<NAME>', c['name'], prompt)
        inputs = tokenizer(s, return_tensors="pt").to("cuda")
        # outputs = model.generate(inputs["input_ids"], do_sample=False, max_new_tokens=1)
        # gender = tokenizer.decode(outputs[0][-1])
        next_logits = model(**inputs)['logits'][:, -1:]
        top10 = torch.topk(next_logits[0, 0], 10)[1].reshape(-1, 1)
        decoded = tokenizer.batch_decode(top10)
        c['gender'] = '?'
        for g in decoded:
            if g.strip() == 'm' or g.strip() == 'f':
                c['gender'] = g.strip()
                break
        characters[idc] = c
    return characters


def get_end_of_sequence_token(model, tokenizer):
    # generation_config = json.load(open(os.path.join(model_path, 'generation_config.json'), 'r'))
    # eos_token_id = int(generation_config["eos_token_id"][0] if isinstance(generation_config["eos_token_id"], list) else generation_config["eos_token_id"])
    eos_token_id = model.config.eos_token_id[0] if isinstance(model.config.eos_token_id, list) else model.config.eos_token_id
    eos_token = tokenizer.decode(eos_token_id)  # This requires tensorflow (for some models?)
    return eos_token


def get_sentences(text, sat_model="models/sat-3l-sm"):
    from wtpsplit import SaT
    sat = SaT(sat_model)
    sat.half().to("cuda")
    sentences = sat.split(text)
    return sentences
