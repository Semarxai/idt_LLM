import requests
import csv
import math
import time
import re
from collections import Counter
from datetime import datetime
import config

import anthropic
from openai import OpenAI
import google.generativeai as genai
from transformers import AutoTokenizer

def clean_phase_label(text):
    import re
    text = str(text)
    text = re.sub(r'\*\*Phase.*?\*\*\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\*\*', '', text)
    return text.strip()


class EntropyEngine:
    def __init__(self):
        print("Loading Llama tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained('NousResearch/Llama-2-7b-hf')
        print("Tokenizer loaded!")
    
    def get_tokens(self, text):
        if not text:
            return []
        return self.tokenizer.encode(text, add_special_tokens=False)
    
    def shannon(self, tokens):
        if not tokens:
            return 0.0
        counts = Counter(tokens)
        total = len(tokens)
        entropy = 0.0
        for c in counts.values():
            p = c / total
            entropy -= p * math.log2(p)
        return entropy

metrics = EntropyEngine()

def calculate_context_entropy(context_text):
    if not context_text:
        return 0.0, 0, 0
    tokens = metrics.get_tokens(context_text)
    entropy = metrics.shannon(tokens)
    unique = len(set(tokens))
    return entropy, len(tokens), unique

def call_teacher_model(provider, prompt):
    provider = provider.lower()
    if provider == "gemini":
        genai.configure(api_key=config.GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-3-pro-preview")
        resp = model.generate_content(prompt)
        return resp.text.strip()
    elif provider == "chatgpt":
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return resp.choices[0].message.content.strip()
    elif provider == "claude":
        client = anthropic.Anthropic(api_key=config.CLAUDE_API_KEY)
        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in resp.content if hasattr(block, "text")).strip()
    else:
        raise ValueError(f"Unknown teacher provider: {provider}")

def call_student_model(messages):
    payload = {
        "model": config.LLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "num_ctx": config.LLAMA_CONTEXT_LIMIT,
            "temperature": config.LLAMA_TEMPERATURE,
            "top_p": config.LLAMA_TOP_P,
            "top_k": config.LLAMA_TOP_K,
            "repeat_penalty": config.LLAMA_REPEAT_PENALTY,
            "num_predict": config.LLAMA_MAX_RESPONSE,
            "seed": config.LLAMA_SEED,
        },
    }
    resp = requests.post(config.OLLAMA_URL, json=payload).json()
    return resp['message']['content']

def setup_csv_files():
    import os
    # Create folder for this teacher
    folder = f"logs_{config.TEACHER_PROVIDER}"
    os.makedirs(folder, exist_ok=True)
    
    metrics_file = f"{folder}/{config.OUTPUT_PREFIX}_metrics.csv"
    with open(metrics_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "DateTime", "Turn",
            "H_S", "H_A", "H_S_prime", "H_SA", "H_SAS_prime",
            "MI_SA_Sprime", "MI_S_A", "P",
            "Hf", "Hb", "Delta",
            "Tokens_S", "Tokens_A", "Tokens_S_prime",
            "Unique_S", "Unique_A", "Unique_S_prime"
        ])
    convo_file = f"{folder}/{config.OUTPUT_PREFIX}_conversation.csv"
    with open(convo_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Turn", "Prompt", "Response"])
    return metrics_file, convo_file

def run_experiment():
    print(f"Starting experiment: {config.OUTPUT_PREFIX}")
    print(f"Teacher: {config.TEACHER_PROVIDER}")
    print(f"Max turns: {config.MAX_TURNS}")
    print("-" * 50)
    
    metrics_file, convo_file = setup_csv_files()
    
    # S = accumulated context (grows each turn)
    S_accumulated = ""
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    
    # Get initial teacher prompt (this is first S'_0, becomes part of S_1)
    teacher_prompt = config.TEACHER_PROMPT + "\n\nStart the conversation now."
    teacher_text = call_teacher_model(config.TEACHER_PROVIDER, teacher_prompt)
    print(f"Initial Teacher: {teacher_text[:100]}...")
    
    # Initialize S with first prompt
    S_accumulated = teacher_text
    messages.append({"role": "user", "content": teacher_text})
    
    for turn in range(1, config.MAX_TURNS + 1):
        try:
            # S = full accumulated context before response
            H_S, tokens_S, unique_S = calculate_context_entropy(S_accumulated)
            
            # A = student response (current turn only)
            student_text = call_student_model(messages)
            H_A, tokens_A, unique_A = calculate_context_entropy(student_text)
            
            print(f"Turn {turn} - Student: {student_text[:100]}...")
            
            # Update messages for next call
            messages.append({"role": "assistant", "content": student_text})
            
            # Get teacher feedback (S' = new prompt only, not accumulated)
            mentor_prompt = (
                f"{config.TEACHER_PROMPT}\n\n"
                f"The Subject just said: {student_text}\n\n"
                f"Continue the conversation. Turn {turn} of {config.MAX_TURNS}."
            )
            teacher_text = call_teacher_model(config.TEACHER_PROVIDER, mentor_prompt)
            
            # S' = teacher feedback only (single prompt)
            S_prime = teacher_text
            H_S_prime, tokens_S_prime, unique_S_prime = calculate_context_entropy(S_prime)
            
            print(f"Turn {turn} - Teacher: {teacher_text[:100]}...")
            
            # Compute H(S,A) - joint entropy of accumulated context + response
            SA_combined = S_accumulated + "\n" + student_text
            H_SA, _, _ = calculate_context_entropy(SA_combined)
            
            # Compute H(S,A,S') - joint entropy of all three
            SAS_prime_combined = S_accumulated + "\n" + student_text + "\n" + S_prime
            H_SAS_prime, _, _ = calculate_context_entropy(SAS_prime_combined)
            
            # Compute metrics
            # MI(S,A;S') = H(S,A) + H(S') - H(S,A,S')
            MI_SA_Sprime = H_SA + H_S_prime - H_SAS_prime
            
            # MI(S;A) = H(S) + H(A) - H(S,A)
            MI_S_A = H_S + H_A - H_SA
            
            # P = MI(S,A;S') / [H(S) + H(A) + H(S')]
            total_H = H_S + H_A + H_S_prime
            P = MI_SA_Sprime / total_H if total_H > 0 else 0.0
            
            # Hf = H(S'|S,A) = H(S,A,S') - H(S,A)
            Hf = H_SAS_prime - H_SA
            
            # Hb = H(S,A|S') = H(S,A,S') - H(S')
            Hb = H_SAS_prime - H_S_prime
            
            # Delta = Hf - Hb
            Delta = Hf - Hb
            
            # Update S_accumulated for next turn (bookkeeping)
            S_accumulated = S_accumulated + "\n[STUDENT]: " + student_text + "\n[TEACHER]: " + teacher_text
            messages.append({"role": "user", "content": teacher_text})
            
            # Log metrics
            with open(metrics_file, "a", newline="") as f:
                csv.writer(f).writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"), turn,
                    H_S, H_A, H_S_prime, H_SA, H_SAS_prime,
                    MI_SA_Sprime, MI_S_A, P,
                    Hf, Hb, Delta,
                    tokens_S, tokens_A, tokens_S_prime,
                    unique_S, unique_A, unique_S_prime
                ])
            
            # Log conversation
            with open(convo_file, "a", newline="") as f:
                csv.writer(f).writerow([turn, clean_phase_label(teacher_text), student_text])
            
            print(f"Turn {turn} complete - P={P:.4f}, Hf={Hf:.4f}, Hb={Hb:.4f}, Delta={Delta:.4f}")
            print("-" * 50)
            time.sleep(1)
            
        except Exception as e:
            print(f"Error at turn {turn}: {e}")
            break
    
    print(f"Experiment complete!")
    print(f"  - {metrics_file}")
    print(f"  - {convo_file}")

if __name__ == "__main__":
    run_experiment()
