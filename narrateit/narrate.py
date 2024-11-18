import os
import time
import re
import json


def format_names(names, n_dialogue=None):
    formatted_output = ""
    if n_dialogue is None:
        for i, name in enumerate(names):
            formatted_output += f"{i + 1}) {name:<15}"
            if (i + 1) % 3 == 0:
                formatted_output += "\n"
    else:
        for i, name in enumerate(names):
            x = name + f", {n_dialogue[i]}"
            formatted_output += f"{i + 1}) {x:<18}"
            if (i + 1) % 3 == 0:
                formatted_output += "\n"
    return formatted_output.strip()


def handle_narration(characters):
    while True:
        if len(characters) > 0:
            x = input(f"Do you want the text to be narrated by an existing character (y/n)?")
        else:
            x = 'n'
        if x == 'y':
            names = list(characters.keys())
            print("Who?")
            print(format_names(names))
            while True:
                choice = input()
                if choice.isdigit() and 1 <= int(choice) <= len(characters):
                    return characters[names[int(choice) - 1]]
                else:
                    print("Invalid choice. Please select a valid number.")
        elif x == 'n':
            narrator = {'name': 'Narrator'}
            while True:
                gender = input(f"Do you want the text to be narrated by a male or female voice (m/f)?")
                if gender in ['m', 'f']:
                    narrator['gender'] = gender
                    break
                else:
                    print("Invalid input. Please enter 'm', or 'f'.")
            while True:
                age = input(f"What should be the approximate age of the narrator (default: 60)?")
                if age.isdigit():
                    narrator['age'] = min(max(int(age), 5), 80)
                else:
                    narrator['age'] = 60
                return narrator
        else:
            print("Invalid input. Please enter 'y', or 'n'.")


def update_character_details(character):
    character_name = character['name']
    while True:
        gender = input(f"Character: {character_name}\nOptions: [male (m)], [female (f)]\n")
        if gender in ['m', 'f']:
            character['gender'] = gender
            break
        else:
            print("Invalid input. Please enter 'm', or 'f'.")
    while True:
        age = input(f"Approximate age of {character_name}:\n")
        if age.isdigit():
            character['age'] = min(max(int(age), 5), 80)
            break
        else:
            print("Invalid input. Please enter a number.")
    return character


def handle_new_character(character_name):
    while True:
        same_character = input(f"Is '{character_name}' always the same character?\nOptions: [no/not sure (n)], [yes (y)]\n")
        if same_character == 'n':
            new_name = input(f"Creating new character, please rename '{character_name}':\n")
            break
        elif same_character == 'y':
            rename = input("Creating new character, do you want to rename? (y/n)\n")
            if rename == 'y':
                new_name = input(f"Please rename '{character_name}':\n")
            else:
                new_name = character_name
            break
        else:
            print("Invalid input. Please enter 'n' or 'y'.")
    return {'name': new_name, 'n_dialogue': 1, 'keep': True}


def handle_old_character(character_name, known_characters):
    print("Who is it?")
    print(format_names(known_characters))
    while True:
        choice = input()
        if choice.isdigit() and 1 <= int(choice) <= len(known_characters):
            chosen_character = known_characters[int(choice) - 1]
            always_same = input(f"Is '{character_name}' always '{chosen_character}'?\nOptions: [Not sure/Only this time (n)], [Yes (y)]\n")
            if always_same in ['n', 'y']:
                return chosen_character, always_same
            else:
                print("Invalid input. Please enter 'n' or 'y'.")
        else:
            print("Invalid choice. Please select a valid number.")


def get_context(annotations, i, max_context=800):
    context = ''
    prefix_length = 0
    if i > 0:
        text = annotations[i - 1]['text']
        if len(text) + prefix_length > max_context:
            text = text[-(max_context - prefix_length):]
        prefix_length += len(text)
        context += annotations[i - 1]['speaker'] + ': ' + text
    if i > 1 and prefix_length + 10 < max_context:
        text = annotations[i - 2]['text']
        if len(text) + prefix_length > max_context:
            text = text[-(max_context - prefix_length):]
        context = annotations[i - 2]['speaker'] + ': ' + text + '\n' + context
    context = '\n"""' + context
    text = annotations[i]['text']
    if len(text) > max_context:
        text = text[:max_context] + '[...]'
    context += '\n' + annotations[i]['speaker'] + ': ' + text
    if i + 1 < len(annotations):
        text = annotations[i + 1]['text']
        if len(text) > max_context:
            text = text[:max_context] + '[...]'
        context += '\n' + annotations[i + 1]['speaker'] + ': ' + text
    context += '\n"""'
    return context


def resolve_characters(document: dict):
    if document.get('characters_resolved', False):
        return document
    characters = document['characters']
    annotations = document['annotation']
    
    n_narration = 0
    if len(characters) > 0:
        # Step 1: Resolve genders
        names = list(characters.keys())
        print("Complete list of annotated characters (might contain dublicates), #dialogue:")
        print(format_names(names, [characters[n]['n_dialogue'] for n in names]))
        print("Please resolve the characters and information.")
        print("Choose option 'n' if you know that this is a different name for another character")
        print("Choose option 'y' if you want to keep this name for the character")
        
        for name in characters:
            while True:
                keep_character = input(
                    f"Character: {name}\nOptions: Keep this character (y), This is a duplicate character (n)\n")
                if keep_character in ['y', 'n']:
                    characters[name]['keep'] = keep_character == 'y'
                    break
                else:
                    print("Invalid input. Please enter 'y' or 'n'.")
        
        # Step 2: Resolve unknown characters in chronological order
        print("Going through the document chronologically, please resolve the unknown characters.")
        known_characters = [name for name in characters if characters[name]['keep']]
        for i in range(len(annotations)):
            character_name = annotations[i]['speaker']
            if any(name == character_name and not characters[name]['keep'] for name in characters):
                context = get_context(annotations, i)
                print(f"Character: {character_name}\nContext: {context}")
                while True:
                    option = input("Options: [New character (n)], [Old character (already classified) (o)]\n")
                    if option == 'n':
                        new_char = handle_new_character(character_name)
                        if new_char['name'] not in list(characters.keys()):
                            characters[new_char['name']] = new_char
                            annotations[i]['speaker'] = new_char['name']
                        else:
                            print("Duplicate name found. Please choose a different name.")
                        break
                    elif option == 'o':
                        old_name, always_same = handle_old_character(character_name, known_characters)
                        if always_same == 'y':
                            for j in range(len(annotations)):
                                if annotations[j]['speaker'] == character_name:
                                    if j < i:
                                        print(f"An earlier instance of '{character_name}' has been found. Replacing all later instances by '{old_name}' but this is probably a mistake.")
                                    else:
                                        annotations[j]['speaker'] = old_name
                                        characters[old_name]['n_dialogue'] += 1
                        else:
                            annotations[i]['speaker'] = old_name
                            characters[old_name]['n_dialogue'] += 1
                        break
                    else:
                        print("Invalid input. Please enter 'n' or 'o'.")
            if character_name == 'Narration':
                n_narration += 1
    
    # Step 3: Resolve Narration and create character dictionary
    print("Please help resolve speaker characteristica for all characters and the narrator")
    time.sleep(3)
    narration = document.get('narration_is_character', '?')
    names = list(characters.keys())
    for name in names:
        if characters[name].pop('keep'):
            characters[name] = update_character_details(characters[name])
        else:
            if narration == name:
                narration = '?'
            characters.pop(name)
    if narration == '?':
        narrator = handle_narration(characters)
        narrator['n_dialogue'] = n_narration
    else:
        narrator = characters[narration]
        narrator['n_dialogue'] += n_narration
    characters['Narration'] = narrator
    document['characters'] = characters
    document['characters_resolved'] = True
    return document


def merge_audio():
    from pydub import AudioSegment
    directory = os.path.join('output', 'waves')
    
    # Regular expression to match the file pattern
    pattern = re.compile(r'out_(\d+)_(\d+).wav')
    files = []
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            # Extract the numbers from the filename
            first_number = int(match.group(1))
            second_number = int(match.group(2))
            files.append((first_number, second_number, filename))
    files.sort(key=lambda x: (x[0], x[1]))
    # sorted_files = [[file[2] for file in files if file[0] == first_number] for first_number in
    #                 sorted(set(file[0] for file in files))]
    
    # Merge audio files
    short_pause_duration, long_pause_duration = 500, 1000
    short_pause = AudioSegment.silent(duration=short_pause_duration)
    long_pause = AudioSegment.silent(duration=long_pause_duration)
    final_audio = AudioSegment.empty()
    previous_first_number = None
    previous_second_number = None
    for first_number, second_number, filename in files:
        audio = AudioSegment.from_wav(os.path.join(directory, filename))
        if previous_first_number is not None:
            if first_number != previous_first_number:
                final_audio += long_pause
            elif second_number != previous_second_number:
                final_audio += short_pause
        final_audio += audio
        previous_first_number = first_number
        previous_second_number = second_number
    
    return final_audio


if __name__ == "__main__":
    config = json.load(open(os.path.join('config', 'config.json')))
    # Make character descriptions
    with open(config['annotation_file'], 'r', encoding='utf-8') as f:
        results = json.load(f)
    results = resolve_characters(results)
    for key in results['characters']:
        print(f"{key}: {results['characters'][key]}")
    for idx in range(len(results['annotation'])):
        print(f"{results['annotation'][idx]['speaker']}: {results['annotation'][idx]['text']}\n")
