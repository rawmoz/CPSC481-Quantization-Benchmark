import requests
import time
import pandas as pd
import re
from datasets import load_dataset

print("Loading dataset...")
data = load_dataset('tinyBenchmarks/tinyGSM8k', 'main')['test']
print(f"Loaded {len(data)} questions")

models = [
    "deepseek-f16",
    "deepseek-q8",
    "deepseek-q6k",
    "deepseek-q4km",
    "deepseek-q2k"
]

def extract_correct_answer(answer_text):
    # Extract number after ####
    match = re.search(r'####\s*([\d,\.]+)', answer_text)
    if match:
        return match.group(1).replace(',', '').strip()
    return None

def format_prompt(question):
    prompt = f"Solve this math problem step by step. At the end, write your final answer as a number on the last line after '####'.\n\n"
    prompt += f"Question: {question}\nAnswer:"
    return prompt

def ask_ollama(model_name, prompt, timeout=120):
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

def extract_predicted_answer(raw_response):
    if raw_response in ["TIMEOUT", "ERROR"]:
        return "X"
    # Strip DeepSeek thinking tags
    cleaned = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL)
    cleaned = cleaned.strip()
    # Look for #### pattern first (model following instructions)
    match = re.search(r'####\s*([\d,\.]+)', cleaned)
    if match:
        return match.group(1).replace(',', '').strip()
    # Fall back to last number in response
    numbers = re.findall(r'[\d,]+\.?\d*', cleaned)
    if numbers:
        return numbers[-1].replace(',', '').strip()
    return "X"

results = []
csv_path = "C:\\CPSC481\\results_gsm8k.csv"

for model in models:
    print(f"\n--- Running model: {model} ---")
    correct = 0

    for i, question in enumerate(data):
        prompt = format_prompt(question['question'])
        correct_answer = extract_correct_answer(question['answer'])

        start_time = time.time()
        raw_response = ask_ollama(model, prompt, timeout=120)
        elapsed = time.time() - start_time

        predicted = extract_predicted_answer(raw_response)
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