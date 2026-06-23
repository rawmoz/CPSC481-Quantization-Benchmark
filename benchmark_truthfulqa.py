import requests
import time
import pandas as pd
import re
from datasets import load_dataset

print("Loading dataset...")
data = load_dataset('tinyBenchmarks/tinyTruthfulQA', 'multiple_choice')['validation']
print(f"Loaded {len(data)} questions")

models = [
    "deepseek-f16",
    "deepseek-q8",
    "deepseek-q6k",
    "deepseek-q4km",
    "deepseek-q2k"
]

def format_prompt(question):
    choices = question['mc1_targets']['choices']
    labels = question['mc1_targets']['labels']
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
    
    prompt = f"Answer this multiple choice question. Reply with only the letter of the correct answer.\n\n"
    prompt += f"Question: {question['question']}\n"
    for i, choice in enumerate(choices):
        prompt += f"{letters[i]}) {choice}\n"
    prompt += "\nAnswer:"
    return prompt

def get_correct_answer(question):
    labels = question['mc1_targets']['labels']
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
    for i, label in enumerate(labels):
        if label == 1:
            return letters[i]
    return None

def ask_ollama(model_name, prompt, timeout=90):
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

def extract_answer(raw_response):
    if raw_response in ["TIMEOUT", "ERROR"]:
        return "X"
    cleaned = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL)
    cleaned = cleaned.strip()
    match = re.search(r'\b([ABCDE])\b', cleaned)
    if match:
        return match.group(1)
    return "X"

results = []
csv_path = "C:\\CPSC481\\results_truthfulqa.csv"

for model in models:
    print(f"\n--- Running model: {model} ---")
    correct = 0

    for i, question in enumerate(data):
        prompt = format_prompt(question)
        correct_answer = get_correct_answer(question)

        start_time = time.time()
        raw_response = ask_ollama(model, prompt, timeout=90)
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
        pd.DataFrame(results).to_csv(csv_path, index=False)

    accuracy = correct / len(data) * 100
    print(f"  {model} accuracy: {accuracy:.1f}%")

print(f"\nAll done! Results saved to {csv_path}")