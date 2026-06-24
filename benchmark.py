import requests ## send requests to Ollama
import json
import time
import pandas as pd
import re
from datasets import load_dataset

# Preparation 
# API CALL (one time) - load the 100 questions, storing all 100 questions into data variable
print("Loading dataset...")
data = load_dataset('tinyBenchmarks/tinyAI2_arc', 'ARC-Challenge')['test']
print(f"Loaded {len(data)} questions")

# Preparation
# 5 models registered in Ollama, storing in models variable
models = [
    "deepseek-f16", 
    "deepseek-q8",
    "deepseek-q6k",
    "deepseek-q4km",
    "deepseek-q2k"
]

# Step 1: Turns questions into readable/formatted question
def format_prompt(question):
    choices = question['choices']
    prompt = f"Answer this multiple choice question. Reply with only the letter A, B, C, or D and nothing else.\n\n"
    prompt += f"Question: {question['question']}\n"
    for label, text in zip(choices['label'], choices['text']):
        prompt += f"{label}) {text}\n"
    prompt += "\nAnswer:"
    return prompt


# Step 2: Sends formatted question --> Ollama and waits for response
    # if it takes longer than 60 seconds we time out
    # where gpu starts to work hard
def ask_ollama(model_name, prompt, timeout=60):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0}

    }
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        result = response.json()
        return result['response']
    except requests.exceptions.Timeout:
        print(f"  TIMEOUT — skipping this question")
        return "TIMEOUT"
    except Exception as e:
        print(f"  ERROR: {e}")
        return "ERROR"


# Step 3: Reads output of step 2 and cleans out the guessed answer
def extract_answer(raw_response):
    if raw_response in ["TIMEOUT", "ERROR"]:
        return "X"
    # Strip DeepSeek thinking tags
    cleaned = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL)
    cleaned = cleaned.strip()
    # Find first occurrence of A, B, C, or D
    match = re.search(r'\b([ABCD])\b', cleaned)
    if match:
        return match.group(1)
    return "X"

# Store all results here
results = []

# CSV path
csv_path = "C:\\CPSC481\\results.csv"

# MAIN
# Step 4: Goes through all 100 questions, CALLING Step1-3 meanwhile...
    # appends a row to results and saves into CSV for every question

# For each of the 5 models...
for model in models:
    print(f"\n--- Running model: {model} ---")
    correct = 0

    # Loop 100 Questions -> inside data from Preperation Stage
    for i, question in enumerate(data):
        # Call Step 1 -> store in prompt
        prompt = format_prompt(question)
        # grab correct answer and store it to compare later
        correct_answer = question['answerKey']

        # Start time
        start_time = time.time()
        # Call Step 2 -> sends question to ollama -> gpu runs -> raw response back
        raw_response = ask_ollama(model, prompt, timeout=60)
        # Stop time -> store in elapsed
        elapsed = time.time() - start_time

        # Call Step 3 -> clean up model output and pull out letter
        predicted = extract_answer(raw_response)
        is_correct = predicted == correct_answer

        # correctness check, if right add one to correct var
        if is_correct:
            correct += 1

        # add new row to CSV
        results.append({
            "model": model,
            "question_id": i,
            "predicted": predicted,
            "correct": correct_answer,
            "is_correct": is_correct,
            "time_seconds": round(elapsed, 2)
        })

        print(f"  Q{i+1}: predicted={predicted}, correct={correct_answer}, time={elapsed:.1f}s")

        # save after every question into CSV
        pd.DataFrame(results).to_csv(csv_path, index=False)

    # after one model is done, get percentage
    accuracy = correct / len(data) * 100
    print(f"  {model} accuracy: {accuracy:.1f}%")

print(f"\nDone. Results saved to {csv_path}")