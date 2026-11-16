import os
import sys
import json
from pathlib import Path
import argparse

from narrateit.annotate import get_model, compare_annotations


def load_jsonl_file(file_path):
    """Load and parse a jsonl file into a list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def process_files(path):
    """Process input path (file or directory) and return a list of all entries."""
    path_obj = Path(path)
    all_entries = []
    
    if path_obj.is_file() and path_obj.suffix.lower() == '.jsonl':
        all_entries.extend(load_jsonl_file(str(path_obj)))
    elif path_obj.is_dir():
        for file_path in path_obj.glob('**/*.jsonl'):
            all_entries.extend(load_jsonl_file(str(file_path)))
    else:
        raise ValueError(f"The provided path '{path}' is not a valid jsonl file or directory containing jsonl files.")
    for entry in all_entries:
        entry['annotation'] = json.loads(entry['annotation'])
    return all_entries


def test_model(model, entries):
    """Test the model against all entries and accumulate results."""
    annotations_correct = []
    words_correct = []
    
    for entry in entries:
        if 'text' not in entry or 'annotation' not in entry:
            print(f"Skipping entry due to missing 'text' or 'annotations': {entry}")
            continue
        
        text = entry['text']
        truth = entry['annotation']['annotation']
        
        # Get model predictions
        generated = model.annotate(text)
        
        # Compare with ground truth
        ann_list, words_list = compare_annotations(truth, generated)
        
        # Accumulate results
        annotations_correct.extend(ann_list)
        words_correct.extend(words_list)
        
        # Print individual results
        n_total = len(ann_list)
        n_correct = sum(ann_list)
        n_words_correct = sum(words_list)
        print(f"Speaker annotations correct {n_correct}/{n_total}")
        print(f"Spoken words that are correct {n_words_correct}/{n_correct}")
        print("________________________________________________")
    
    return annotations_correct, words_correct


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a model against annotated data in jsonl files.")
    parser.add_argument("path", help="Path to a jsonl file or a directory containing jsonl files")
    args = parser.parse_args()
    
    # Process files and get all entries
    all_entries = process_files(args.path)
    print(f"Loaded {len(all_entries)} entries from the provided path.")

    if not all_entries:
        print("No valid entries found. Exiting.")
        sys.exit(1)

    # Test model on all entries
    model = get_model()
    all_pass_results, all_words_results = test_model(model, all_entries)

    # Calculate and print final statistics
    total_pass = sum(all_pass_results)
    total_words = sum(all_words_results)
    n_total = len(all_pass_results)

    print("\n=== FINAL RESULTS ===")
    print(f"Annotations tests: {total_pass}/{n_total} ({total_pass / n_total:.2%})")
    print(f"Words tests: {total_words}/{n_total} ({total_words / n_total:.2%})")
