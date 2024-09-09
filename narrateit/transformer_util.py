import os
import json
import re
import time
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList

from .util import write_to_file

torch.random.manual_seed(0)


def load_model_tokenizer(model_path):
    print(f"Loading model '{model_path}'...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.appended_token = int(tokenizer('}', return_tensors='pt').to('cuda')['input_ids'][0][-1])
    tokenizer.curly1 = int(tokenizer('{', return_tensors='pt').to('cuda')['input_ids'][0][-1])
    tokenizer.curly2 = int(tokenizer('}', return_tensors='pt').to('cuda')['input_ids'][0][-1])
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
    
    outputs = model.generate(inputs["input_ids"],
                             do_sample=False,
                             max_new_tokens=max_new_tokens,
                             stopping_criteria=stopping_criteria)
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


def generate_based_on_new_state(prompt, model, tokenizer, max_new_tokens=200):
    """Doesn't work with peft"""
    pass


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

# prompt = 'User: How many people live in France?\nAssistant: '
# inputs = tokenizer(prompt, return_tensors='pt').to("cuda")
# output1 = model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True, cache_implementation=None, do_sample=False)
# sequence = tokenizer.batch_decode(output1.sequences)[0]
# print(sequence)
# prompt = sequence + "7 million people live in France. \n<end_of_turn>\n User: And how many in Germany?\nAssistant: "
# inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
# generation_output2 = model.generate(**inputs, past_key_values=output1.past_key_values, max_new_tokens=30, return_dict_in_generate=True, cache_implementation=None, do_sample=False)

