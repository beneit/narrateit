import os, json
import torch
from peft import get_peft_model, PromptTuningConfig, TaskType, PromptTuningInit, PeftModel, PrefixTuningConfig
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling, default_data_collator, get_linear_schedule_with_warmup, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import transformers
from typing import Optional
import warnings
import packaging.version
from peft.utils import PeftType
from peft.peft_model import PeftModelForCausalLM
from .util import config
from .transformer_util import load_model_tokenizer, load_tokenizer


def create_peft_model(model, num_virtual_tokens=10, initial_prompt=None, prefix_tuning=False):
    # if initial_prompt is None:
    #     initial_prompt = ""
    prompt_tuning_init = PromptTuningInit.RANDOM if initial_prompt is None else PromptTuningInit.TEXT
    tokenizer_name_or_path = config['annotation_model'] if initial_prompt is None else None
    if prefix_tuning:
        peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=num_virtual_tokens)
    else:
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=prompt_tuning_init,
            # prompt_tuning_info=PromptTuningInit.TEXT,
            prompt_tuning_init_text=initial_prompt,
            num_virtual_tokens=num_virtual_tokens,
            tokenizer_name_or_path=config['annotation_model']
        )
    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()
    return peft_model


def create_training_arguments(path, learning_rate=0.0035, epochs=6):
    training_args = TrainingArguments(
        output_dir=path,  # Where the model predictions and checkpoints will be written
        # use_cpu=True,  # This is necessary for CPU clusters.
        auto_find_batch_size=True,  # Find a suitable batch size that will fit into memory automatically
        learning_rate=learning_rate,  # Higher learning rate than full fine-tuning
        num_train_epochs=epochs)
    return training_args


def preprocess_function(examples):
    text_column = 'text'
    label_column = 'annotation'
    tokenizer = load_tokenizer(config['annotation_model'])
    batch_size = len(examples[text_column])
    inputs = [f"input: {x}\noutput : " for x in examples[text_column]]
    targets = [str(x) for x in examples[label_column]]
    model_inputs = tokenizer(inputs)
    labels = tokenizer(targets)
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i] + [tokenizer.pad_token_id]
        model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
        labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
    max_length = max([len(model_inputs["input_ids"][i]) for i in range(batch_size)]) + 0
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]
        diff = max_length - len(sample_input_ids)
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * diff + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * diff + model_inputs["attention_mask"][i]
        labels["input_ids"][i] = [-100] * diff + label_input_ids
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
        labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def train_peft(peft_model, dataset, output_directory=None, num_epochs=1, learning_rate=0.05):
    dataset = load_dataset(dataset)
    # data = data.map(lambda samples: tokenizer(samples["text"]), batched=True)
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=1,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )
    train_sample = processed_dataset["train"]
    # eval_sample = processed_dataset["test"]
    # train_sample = data["train"].select(range(11)
    # print(train_sample)
    batch_size = 1  # len(processed_dataset["train"])
    train_dataloader = DataLoader(train_sample, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size,
                                  pin_memory=True)
    # eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)
    eval_dataloader = train_dataloader

    # training_args = create_training_arguments(output_directory_prompt, 0.003, NUM_EPOCHS)
    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=learning_rate)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )
    
    # def create_trainer(peft_model, training_args, train_dataset):
    #     trainer = Trainer(
    #         model=peft_model, # We pass in the PEFT version of the foundation model
    #         args=training_args, #The args for the training.
    #         train_dataset=train_dataset, #The dataset used to train the model.
    #         data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False) # mlm=False indicates not to use masked language modeling
    #         )
    #     return trainer
    
    device = 'cuda'
    peft_model = peft_model.to(device)
    
    for epoch in range(num_epochs):
        peft_model.train()
        total_loss = 0
        try:
            for step, batch in enumerate(tqdm(train_dataloader)):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = peft_model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
        except KeyboardInterrupt:
            break
        
        # peft_model.eval()
        # eval_loss = 0
        # eval_preds = []
        # for step, batch in enumerate(tqdm(eval_dataloader)):
        #     batch = {k: v.to(device) for k, v in batch.items()}
        #     with torch.no_grad():
        #         outputs = peft_model(**batch)
        #     loss = outputs.loss
        #     eval_loss += loss.detach().float()
        #     eval_preds.extend(
        #         tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(),
        #                                skip_special_tokens=True)
        #     )
        # 
        # eval_epoch_loss = eval_loss / len(eval_dataloader)
        # eval_ppl = torch.exp(eval_epoch_loss)
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        # print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")
        print(f"{epoch=}: {train_ppl=} {train_epoch_loss=}")
    
    # trainer_prompt = create_trainer(peft_model, training_args, train_sample)
    # trainer_prompt.train()
    
    if output_directory is None:
        dataset_name = 'examples'
        peft_config = peft_model.peft_config['default']
        base = peft_config.base_model_name_or_path.split('/')[-1]
        output_directory = f"{dataset_name}_{base}_{peft_config.peft_type.value}".replace("/", "_")
        output_directory = os.path.join('peft_model', output_directory)
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    # trainer_prompt.model.save_pretrained(output_directory_prompt)
    peft_model.save_pretrained(output_directory)
    
    # loaded_model = PeftModel.from_pretrained(
    #     model,
    #     output_directory,
    #     # device_map='auto',
    #     is_trainable=False,
    # )
    # loaded_model_prompt_outputs = get_outputs(loaded_model_prompt, input_prompt)
    # print(tokenizer.batch_decode(loaded_model_prompt_outputs, skip_special_tokens=True))


def load_peft_model(model, model_path):
    from peft import PeftModel, PeftConfig
    
    working_dir = "peft_model/"
    peft_model_id = os.path.join(working_dir, model_path)
    
    # peft_config = PeftConfig.from_pretrained(peft_model_id)
    peft_model = PeftModel.from_pretrained(model, peft_model_id)
    return peft_model


class PeftModelForCausalLM_mod(PeftModelForCausalLM):
    def prepare_inputs_for_generation(self, *args, task_ids: Optional[torch.Tensor] = None, **kwargs):
        peft_config = self.active_peft_config
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)
        
        # https://github.com/huggingface/transformers/pull/26681/ introduced new cache format
        # for some architectures which requires a special fix for prompt tuning etc.
        # TODO: starting with transformers 4.38, all architectures should support caching.
        uses_transformers_4_38 = packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.38.0")
        uses_transformers_4_36 = packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.36.0")
        transformers_new_cache_archs = ["llama", "mistral", "persimmon", "phi"]
        if packaging.version.parse(transformers.__version__) > packaging.version.parse("4.43.3"):
            # https://github.com/huggingface/transformers/pull/31445
            transformers_new_cache_archs.append("bloom")
    
        uses_cache = uses_transformers_4_38 or (
                uses_transformers_4_36 and self.base_model.config.model_type in transformers_new_cache_archs
        )
        
        if peft_config.peft_type == PeftType.POLY:
            model_kwargs["task_ids"] = task_ids
        if peft_config.is_prompt_learning:
            if uses_cache and (model_kwargs["past_key_values"] is not None):
                # change in the logic of `prepare_inputs_for_generation` makes the below code necessary
                # In prompt learning methods, past key values are longer when compared to the `input_ids`.
                # As such only consider the last input ids in the autogressive generation phase.
                past_key_values = model_kwargs["past_key_values"]
                if isinstance(past_key_values, (tuple, list)):
                    seq_len = past_key_values[0][0].shape[-2]
                else:  # using transformers kv cache
                    seq_len = past_key_values.get_seq_length()
                if seq_len is None:
                    seq_len = peft_config.num_virtual_tokens + model_kwargs["input_ids"].shape[1]
                if seq_len >= model_kwargs["input_ids"].shape[1]:
                    model_kwargs["input_ids"] = model_kwargs["input_ids"][:, -1:]
            
            if model_kwargs.get("attention_mask", None) is not None:
                size = model_kwargs["input_ids"].shape[0], peft_config.num_virtual_tokens
                prefix_attention_mask = torch.ones(size).to(model_kwargs["input_ids"].device)
                model_kwargs["attention_mask"] = torch.cat(
                    (prefix_attention_mask, model_kwargs["attention_mask"]), dim=1
                )
            
            if model_kwargs.get("position_ids", None) is not None:
                warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
                model_kwargs["position_ids"] = None
            
            if kwargs.get("token_type_ids", None) is not None:
                warnings.warn(
                    "Token type ids are not supported for parameter efficient tuning. Ignoring token type ids"
                )
                kwargs["token_type_ids"] = None
            
            # no past_key_values or past_key_values empty cache
            requires_prompt_injection = (model_kwargs["past_key_values"] is None) or (
                    isinstance(model_kwargs["past_key_values"], transformers.Cache) and not model_kwargs[
                "past_key_values"]
            )
            
            if requires_prompt_injection and peft_config.peft_type == PeftType.PREFIX_TUNING:
                new_past_key_values = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0])
                model_kwargs["past_key_values"] = new_past_key_values
            elif requires_prompt_injection:
                inputs_embeds = self.word_embeddings(model_kwargs["input_ids"])
                prompts = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0], task_ids=task_ids)
                prompts = prompts.to(inputs_embeds.dtype)
                model_kwargs["inputs_embeds"] = torch.cat((prompts, inputs_embeds), dim=1)
                model_kwargs["input_ids"] = None
        
        # For transformers>=4.38.0 - for some architectures such as Llama, `cache_position` is
        # passed in the forward pass to keep track of the position ids of the cache. We have to
        # pop that from `model_kwargs` as `cache_position` is properly created by the model, using the passed
        # `inputs_embeds`: https://github.com/huggingface/transformers/blob/593230f0a1150ea9c0477b9d859f25daf73c8c33/src/transformers/models/llama/modeling_llama.py#L956
        _ = model_kwargs.pop("cache_position", None)
        
        return model_kwargs


if __name__ == "__main__":
    num_virtual_tokens = 10
    num_epochs = 100
    learning_rate = 0.05
    
    dataset = "datasets/annotation"
    model_path = config['annotation_model']
    model, tokenizer = load_model_tokenizer(model_path)
    peft_model = create_peft_model(model, num_virtual_tokens=num_virtual_tokens)
    train_peft(peft_model, dataset, num_epochs=num_epochs, learning_rate=learning_rate)
