import openai

# Initialize the API key
openai.api_key = "your-api-key"

# Step 1: Provide task explanation with guidelines and examples
task_explanation = [
    {"role": "system", "content": "You are an assistant trained to analyze literary excerpts. Your job is to: "
                                  "1. Provide reasoning about which character, if any, speaks in the given text. "
                                  "2. Classify the text as 'Narration', 'Character', or 'Both'. "
                                  "3. If 'Character' is selected, specify the name of the character speaking."},
    {"role": "user", "content": "Here are some examples of how you should respond:\n\n"
                                "Example 1:\n"
                                "Text before section: 'Frodo hesitated and then replied.'\n"
                                "Section to analyze: 'Of course I trust you, Gandalf.'\n"
                                "Text after section: 'But his eyes showed doubt.'\n"
                                "Reasoning: The words in the section contain quotation marks and are framed as dialogue. "
                                "The speaker is likely Frodo, based on the context before and after.\n"
                                "Output: Character\n"
                                "Name: Frodo\n\n"
                                "Example 2:\n"
                                "Text before section: 'Gollum slunk back into the shadows.'\n"
                                "Section to analyze: 'The cave was cold and dark.'\n"
                                "Text after section: 'He shivered and muttered to himself.'\n"
                                "Reasoning: The section describes the setting and does not contain dialogue. It is narration.\n"
                                "Output: Narration\n"
                                "Name: None"}
]

# Step 2: First query - Ask for reasoning
query_1 = [
    {"role": "user", "content": "Text before section: 'Sam glanced at the sky nervously.'\n"
                                "Section to analyze: 'I think we should keep moving.'\n"
                                "Text after section: 'He shifted his pack and sighed.'\n"
                                "What reasoning leads to the correct classification?"}
]

response_1 = openai.ChatCompletion.create(
    model="gpt-4",
    messages=task_explanation + query_1,
    temperature=0.7  # Adjust creativity if needed
)
reasoning = response_1.choices[0].message["content"]
print("Reasoning:", reasoning)

# Step 3: Second query - Ask for classification (one-word answer)
query_2 = [
    {"role": "user", "content": "Based on the reasoning, classify the section as Narration, Character, or Both."}
]

response_2 = openai.ChatCompletion.create(
    model="gpt-4",
    messages=task_explanation + query_1 + [{"role": "assistant", "content": reasoning}] + query_2,
    temperature=0.0  # Deterministic output for classification
)
classification = response_2.choices[0].message["content"]
print("Classification:", classification)

# Step 4: Third query - Ask for name of the character, if applicable
if classification.strip() == "Character":
    query_3 = [
        {"role": "user", "content": "What is the name of the character speaking in the section?"}
    ]

    response_3 = openai.ChatCompletion.create(
        model="gpt-4",
        messages=task_explanation + query_1 + [{"role": "assistant", "content": reasoning}] +
                 [{"role": "assistant", "content": classification}] + query_3,
        temperature=0.7  # Allow creativity for naming the character
    )
    character_name = response_3.choices[0].message["content"]
    print("Name:", character_name)
else:
    print("No character name required.")
