import requests
import json
import time
import pandas as pd
import re
from datasets import load_dataset

# Load the 100 questions
print("Loading dataset...")
data = load_dataset('tinyBenchmarks/tinyAI2_arc', 'ARC-Challenge')['test']
print(f"Loaded {len(data)} questions")

# The 5 models we registered in Ollama
models = [
    "deepseek-f16",
    "deepseek-q8",
    "deepseek-q6k",
    "deepseek-q4km",
    "deepseek-q2k"
]

def format_prompt(question):
    choices = question['choices']
    prompt = f"Answer this multiple choice question. Reply with only the letter A, B, C, or D and nothing else.\n\n"
    prompt += f"Question: {question['question']}\n"
    for label, text in zip(choices['label'], choices['text']):
        prompt += f"{label}) {text}\n"
    prompt += "\nAnswer:"
    return prompt

def ask_ollama(model_name, prompt, timeout=60):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
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

for model in models:
    print(f"\n--- Running model: {model} ---")
    correct = 0

    for i, question in enumerate(data):
        prompt = format_prompt(question)
        correct_answer = question['answerKey']

        start_time = time.time()
        raw_response = ask_ollama(model, prompt, timeout=60)
        elapsed = time.time() - start_time

        predicted = extract_answer(raw_response)
        is_correct = predicted == correct_answer

        if is_correct:
            correct += 1

        results.append({
            "model": model,
            "question_id": i,
            "predicted": predicted,
            "correct": correct_answer,
            "is_correct": is_correct,
            "time_seconds": round(elapsed, 2)
        })

        print(f"  Q{i+1}: predicted={predicted}, correct={correct_answer}, time={elapsed:.1f}s")

        # Save after every question
        pd.DataFrame(results).to_csv(csv_path, index=False)

    accuracy = correct / len(data) * 100
    print(f"  {model} accuracy: {accuracy:.1f}%")

print(f"\nAll done! Results saved to {csv_path}")