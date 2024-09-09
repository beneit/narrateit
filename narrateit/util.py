import re
import json
import os

config = json.load(open(os.path.join('config', 'config.json')))


def write_to_file(filename, text):
    with open(f"{filename}.txt", "w", encoding="utf-8") as file:
        file.write(text)


def clean_annotations(annotations):
    """
    merge subsequent annotations with the same speaker.
    annotations is a list with dict entries. Each dict has 'speaker' and 'text'. If two subsequent speakers are
    identical, merge the text.
    """
    _annotations = []
    for annotation in annotations:
        if len(_annotations) == 0 or _annotations[-1]['speaker'] != annotation['speaker']:
            _annotations.append(annotation.copy())
        else:
            _annotations[-1]['text'] += ' ' + annotation['text']
    return _annotations


def clean_string(s):
    s = '\n'.join(line.strip() for line in s.splitlines())
    s = re.sub(r'[ \t]+', ' ', s)
    s = re.sub(r'(\r?\n)+', '\n', s)
    s = re.sub(r'[“”]', '"', s)
    s = re.sub(r'[‘’]', "'", s)
    s = s.replace('{', '(').replace('}', ')')
    return s


def save_json(filename, s):
    with open(filename, 'w', encoding='utf-8') as json_file:
        json.dump(s, json_file, indent=4, ensure_ascii=False)


def get_text_from_pdf(filename):
    from pypdf import PdfReader
    reader = PdfReader(filename)
    num_pages = len(reader.pages)
    text = ''
    for i in range(num_pages):
        page = reader.pages[i]
        text += page.extract_text()
    return text


# def old_get_annotation(initial_prompt, characters, last_characters, last_genders,
#                    text, overlap, previous_text, previous_annotations, replacements):
#     def extract_from_text(s):
#         try:
#             return json.loads(s)["annotation"]
#         except json.decoder.JSONDecodeError:  # TODO: Save program state (somehow)
#             print("Error in LLM output format, illegal json. Aborting...")
#             raise BadLLMOutput(f"Bad LLM output: \n######{s}\n######")
# 
#     new_text = get_generated(prompt, model, tokenizer, len(text))
#     write_to_file(f"inputs_plus_outputs_tmp/prompt_plus_generated_{int(time.time())}",
#                   prompt + new_text)  # TODO: DELETE THIS
#     annotations = extract_from_text("{\"annotation\":[" + new_text)
#     annotations = clean_annotations(annotations)
#     annotations_overlap = []
#     if len(overlap) > 0:
#         prompt = append_annotations(prompt, annotations)
#         prompt += eos_token + "\n\nText: " + overlap + "\n\noutput: {\n    \"annotation\": [\n"
#         new_text = get_generated(prompt, model, tokenizer, len(overlap))
#         write_to_file(f"inputs_plus_outputs_tmp/prompt_plus_generated_{int(time.time())}",
#                       prompt + new_text)  # TODO: DELETE THIS
#         annotations_overlap = extract_from_text("{\"annotation\":[" + new_text)
#     return annotations, annotations_overlap

def old_append_annotations(prompt, annotations):
    """
    This assumes prompt ends with: "    "annotation": ["
    Will stop with "}"
    """
    prompt += "\n"
    for i, annotation in enumerate(annotations):
        prompt += "        {\n"
        prompt += "            \"speaker\": \"" + annotation['speaker'] + "\",\n"
        prompt += "            \"text\": \"" + annotation['text'] + "\"\n"
        prompt += "        }"
        if i + 1 < len(annotations):
            prompt += ","
        prompt += "\n"
    prompt += "    ]\n"
    prompt += "}"
    return prompt


def old_create_dual_segments(sentences, min_segment_length, min_overlap_length):
    segments = []
    idx_start = 0
    idx_end = 0
    segment_lengths = (min_segment_length, min_overlap_length)
    swap = True
    min_segment_length_ = min_segment_length
    for j, sentence in enumerate(sentences):
        idx_end += len(sentence)
        if idx_end - idx_start > min_segment_length_:
            segments.append([idx_start, idx_end])
            idx_start = idx_end
            min_segment_length_ = segment_lengths[swap]
            swap = not swap
    segments.append([idx_start, idx_end])
    return segments
