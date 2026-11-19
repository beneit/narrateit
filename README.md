# NarrateIt
### Table of Contents
- [Installation](#installation)
- [Annotation](#annotation)
- [Character-Voice Mapping](#character-voice-mapping)
- [Narration](#narration)

### Description
- Convert a book to an audiobook using character specific narration.
- Text is annotated using an LLM. Supports huggingface text-to-text (causal language) models
- Annotated text is converted to audio using neural speech synthesis
- Static Demo of the webapp using processed data: https://beneit.github.io/narrateit/


### Installation
```bash
git clone https://github.com/beneit/narrateit.git
cd narrateit
pip install -e .
```
### Annotation
The annotation of a narrative text proceeds by utilizing an
LLM in a structured, few-shot approach.
Going through the text line by line, sentence by sentence
(and potentially word by word) each section is classified 
whether it be spoken by the Narrator or any particular 
character. I am currently using Llama-3.2-3B-Instruct as the LLM
and XTTS-v2 as the voice model which means that the environment
needs both of them running (or two seperate environments, splitting the process)
Edit the config:
```
config/config.json
```
to point to your document (txt file) and your model.
Edit the "annotation_model" to point to your model 
(transformer) or the name of the model to download from 
huggingface. Put "openai/name_of_model" to use openAI API 
models.
```
python main.py --annotate
```
This will create the annotation file in:
```
output/mytext/annotations.json
```
Depending on the capabilities of the LLM, results will vary.
For now, the prompting technique is still in the refining stage
and I am gathering data to create prompt-tuned models using 
[peft](https://github.com/huggingface/peft) 
(I am currently working on the functionality)

Check the file for errors. Depending on error rate and the 
length of the document, you will be able to use that file 
for further processing.
### Character-Voice Mapping
For now, characters are not automatically "merged"
(different names might be given to the same character 
throughout the length of the document) or characterized by 
gender and age. Voice characteristics are also not
made available from the text to influence tts models
(such functionality is still in its early stages anyways)
```
python main.py --characters
```
Enter the relevant information about the characters to the 
best of your knowledge. This information is used for the
character-voice mappings.
This will edit your input into the annotation file created during the
annotation step.

### Narration
The final step uses a tts model (xtts) to read in the lines based
on the annotations.json file.
```
python main.py --narrate
```
Final result is postprocessed and merged to a single file:
```
output/mytext/mytext.mp3
```