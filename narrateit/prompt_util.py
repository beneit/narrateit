import re
import json


def create_segments(text, min_segment_length, min_overlap_length):
    # Split the text into sentences using the end of sentence indicators ".", "?", and "!"
    sentences = re.split(r'(?<=[.!?])', text)
    
    segments = []
    idx_start = 0
    idx_end = 0
    idx_next_start = 0
    for j, sentence in enumerate(sentences):
        idx_end += len(sentence)
        if idx_end - idx_start > min_segment_length and idx_next_start == idx_start:
            idx_next_start = idx_end
        if idx_end - idx_start > min_segment_length + min_overlap_length and \
                idx_end >= idx_next_start + min_overlap_length:
            segments.append([idx_start, idx_end])
            idx_start = idx_next_start
    segments.append([idx_start, idx_end])
    return segments


def create_dual_segments(sentences, min_segment_length, min_overlap_length):
    segments = []
    segment = ''
    idx_start = 0
    idx_end = 0
    segment_lengths = (min_segment_length, min_overlap_length)
    swap = True
    min_segment_length_ = min_segment_length
    for j, sentence in enumerate(sentences):
        idx_end += len(sentence)
        segment += sentence
        if idx_end - idx_start > min_segment_length_:
            segments.append(segment)
            segment = ''
            idx_start = idx_end
            min_segment_length_ = segment_lengths[swap]
            swap = not swap
    segments.append(segment)
    return segments


def get_prompt(initial_prompt, all_speakers=None, last_speakers=None, genders=None, replacements=None):
    last_speakers_ = [] if last_speakers is None else last_speakers.copy()
    if len(last_speakers_) == 0:
        last_speakers_.append("Lennie")
        genders_ = ['m']
    else:
        genders_ = genders.copy()
    if len(last_speakers_) == 1:
        last_speakers_.append("George")
        genders_.append('m')
    all_speakers_ = [] if all_speakers is None else all_speakers.copy()
    list_of_speakers = ""
    for a_speaker in all_speakers_:
        if a_speaker not in last_speakers_:
            list_of_speakers += "\n" + a_speaker
    
    def create_pronoun_dict(g):
        pronouns = {
            'm': {
                'GENDER': 'man',
                'SUBJECT_PRONOUN': 'he',
                'OBJECT_PRONOUN': 'him',
                'POSSESSIVE_ADJ': 'his',
                'POSSESSIVE_PRONOUN': 'his',
                'REFLEXIVE_PRONOUN': 'himself',
                'HONORIFIC': 'Mr.'
            },
            'f': {
                'GENDER': 'woman',
                'SUBJECT_PRONOUN': 'she',
                'OBJECT_PRONOUN': 'her',
                'POSSESSIVE_ADJ': 'her',
                'POSSESSIVE_PRONOUN': 'hers',
                'REFLEXIVE_PRONOUN': 'herself',
                'HONORIFIC': 'Ms.'
            },
            'd': {
                'GENDER': 'person',
                'SUBJECT_PRONOUN': 'they',
                'OBJECT_PRONOUN': 'them',
                'POSSESSIVE_ADJ': 'their',
                'POSSESSIVE_PRONOUN': 'theirs',
                'REFLEXIVE_PRONOUN': 'themselves',
                'HONORIFIC': 'Mx.'
            }
        }
        result = {}
        assert (len(g) == 2)
        for i, gender in enumerate(g, 1):
            for key, value in pronouns[gender].items():
                result[f'{key}_{i}'] = value
                if key in ['GENDER', 'SUBJECT_PRONOUN', 'OBJECT_PRONOUN', 'POSSESSIVE_ADJ', 'POSSESSIVE_PRONOUN']:
                    result[f'{key}_CAP_{i}'] = value.capitalize()
        return result
    
    replacements.update({
        "Lennie": last_speakers_[0],
        "George": last_speakers_[1],
        "LIST_OF_SPEAKERS": list_of_speakers
    })
    replacements.update(create_pronoun_dict(genders_))
    
    def replace_keyword(match):
        keyword = match.group(1)
        return replacements.get(keyword, match.group(0))
    
    prompt = re.sub(r'<(.*?)>', replace_keyword, initial_prompt)
    return prompt


def append_prompt(prompt, text, previous_annotations):
    """
    special cases: 1. previous_annotations are empty, last: text only has text from previous annotations
    """
    prompt = re.sub(r'<TEXT>', text, prompt)
    prompt += " {\n    \"annotation\": ["
    for i, annotation in enumerate(previous_annotations):
        prompt += "\n        {\n"
        prompt += "            \"speaker\": \"" + annotation['speaker'] + ",\n"
        prompt += "            \"text\": \"" + annotation['text'] + "\"\n"
        prompt += "        }"
        if i + 1 < len(previous_annotations):
            prompt += ","
    return prompt


def append_previous_annotations(annotations):
    """
    special cases: 1. annotations are empty, last: text only has text from previous annotations
    """
    s = "{\n    \"annotation\": [\n"
    for i, annotation in enumerate(annotations):
        s += "        {\n"
        s += "            \"speaker\": \"" + annotation['speaker'] + ",\n"
        s += "            \"text\": \"" + annotation['text'] + "\"\n"
        s += "        }"
        s += ",\n"
    return s


def append_annotations(prompt, old_annotations):
    """
    This assumes prompt ends with: %    "annotation": [%
    Will stop with: %            "speaker": "%
    """
    prompt += '\n        {\n            "speaker": "'
    for i, annotation in enumerate(old_annotations):
        prompt += annotation['speaker'] + '",\n'
        prompt += '            "text": "' + annotation['text']
        # if i + 1 < len(old_annotations):
        prompt += '"\n        },\n        {\n            "speaker": "'
    return prompt


def create_data_set(file, output_file):
    """
    file is a json. Top level object is a list.
        Each entry is a dict. Each dict has two entries: "text", "annotation".
            "text" value is a string
            "annotation" value is a list. Entries of the list are dicts
                Each dict has two entries: "speaker", "text".
                    "speaker" value is a string
                    "text" value is a string
    This function saves a jsonl file
    Each line of the output is one of the entries of the input file (which was a list)
    Each line is an object with two fields, "text", "annotation"
        "text" value is a string, the same as in original
        "annotation" value is a string. This string is a formatted json like so:
    "{\n    \"annotation\": [\n        {\"speaker\": \"val_speaker\", \"text\": \"val_text\"},\n      {...}\n    ]\n}
    Here, {...} is another list entry of the "annotation" key of an input file list element.
    val_speaker is the value of the speaker field and val_text is the value of the text field.
    """
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for entry in data:
            text = entry['text']
            annotation = entry['annotation']
            annotation_str = json.dumps({"annotation": annotation}, ensure_ascii=False, indent=4)
            jsonl_entry = {
                "text": text,
                "annotation": annotation_str
            }
            f_out.write(json.dumps(jsonl_entry, ensure_ascii=False) + '\n')
