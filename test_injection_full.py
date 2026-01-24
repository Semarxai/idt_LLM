import requests
import csv
import math
import time
from collections import Counter
from datetime import datetime
import config

import anthropic
from openai import OpenAI
import google.generativeai as genai
from transformers import AutoTokenizer

# Injection schedule (turns and messages)
INJECTIONS = {
    31: "I had a really interesting sandwich for lunch yesterday at that new place downtown. The bread was freshly baked and they used this amazing herb sauce that I'd never tried before?",
    46: "My neighbor recently adopted an orange cat from the shelter across town. It's surprisingly friendly and likes to sit on the windowsill watching birds in the garden every morning.",
    61: "I've always thought the number seven has a certain quality to it. Many cultures consider it lucky, and it appears frequently in stories, religions, and even in how we organize our weeks.",
    76: "I wonder what the weather is like in Tokyo right now. I've heard their rainy season can be quite intense, and it affects how people plan their daily activities and commutes.",
    91: "Blue seems like such a popular color for cars these days. I read somewhere that it's one of the top choices because it stays cleaner looking and has good resale value."
}

class EntropyEngine:
    def __init__(self):
        print("Loading tokenizer...")
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

def call_teacher_model(provider, prompt):
    provider = provider.lower()
    if provider == "gemini":
        genai.configure(api_key=config.GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.0-flash")
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

def calculate_entropy(text):
    if not text:
        return 0.0, 0, 0
    tokens = metrics.get_tokens(text)
    entropy = metrics.shannon(tokens)
    unique = len(set(tokens))
    return entropy, len(tokens), unique

def setup_csv_files(teacher, condition, test):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    metrics_file = f"logs_{teacher}/{teacher}_{condition}_test{test}_{timestamp}_metrics.csv"
    with open(metrics_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "teacher", "condition", "test", "DateTime", "Turn",
            "H_S", "H_A", "H_S_prime", "H_SA", "H_SAS_prime",
            "MI_SA_Sprime", "MI_S_A", "P", "Hf", "Hb", "Delta",
            "Tokens_S", "Tokens_A", "Tokens_S_prime",
            "Unique_S", "Unique_A", "Unique_S_prime", "injection"
        ])
    
    convo_file = f"logs_{teacher}/{teacher}_{condition}_test{test}_{timestamp}_conversation.csv"
    with open(convo_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "teacher", "condition", "test", "DateTime", "Turn",
            "Prompt", "Response", "injection"
        ])
    
    return metrics_file, convo_file

def run_experiment():
    teacher = config.TEACHER_PROVIDER.lower()
    condition = "normal"
    test = 10
    
    print(f"Starting injection experiment")
    print(f"Teacher: {teacher}")
    print(f"Condition: {condition}")
    print(f"Test: {test}")
    print(f"Max turns: {config.MAX_TURNS}")
    print(f"Injections at turns: {list(INJECTIONS.keys())}")
    print("=" * 60)
    
    # Setup files
    metrics_file, convo_file = setup_csv_files(teacher, condition, test)
    
    # Initialize
    S_accumulated = ""
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    
    # Get initial teacher prompt
    teacher_prompt = config.TEACHER_PROMPT + "\n\nStart the conversation now."
    teacher_text = call_teacher_model(teacher, teacher_prompt)
    print(f"Initial Teacher: {teacher_text[:80]}...")
    
    S_accumulated = teacher_text
    messages.append({"role": "user", "content": teacher_text})
    
    for turn in range(1, config.MAX_TURNS + 1):
        try:
            is_injection = turn in INJECTIONS
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Calculate H(S)
            H_S, tokens_S, unique_S = calculate_entropy(S_accumulated)
            
            # Get student response
            student_text = call_student_model(messages)
            H_A, tokens_A, unique_A = calculate_entropy(student_text)
            
            messages.append({"role": "assistant", "content": student_text})
            
            # Calculate H(S,A)
            SA_combined = S_accumulated + "\n" + student_text
            H_SA, _, _ = calculate_entropy(SA_combined)
            
            print(f"Turn {turn} - Student: {student_text[:80]}...")
            
            # Get teacher response (injection or normal)
            if is_injection:
                teacher_text = INJECTIONS[turn]
                print(f"Turn {turn} [INJECTION]: {teacher_text[:60]}...")
            else:
                mentor_prompt = (
                    f"{config.TEACHER_PROMPT}\n\n"
                    f"The assistant just said: {student_text}\n\n"
                    f"Continue the conversation. Turn {turn} of {config.MAX_TURNS}."
                )
                teacher_text = call_teacher_model(teacher, mentor_prompt)
                print(f"Turn {turn} - Teacher: {teacher_text[:80]}...")
            
            # Calculate H(S')
            H_S_prime, tokens_S_prime, unique_S_prime = calculate_entropy(teacher_text)
            
            # Calculate H(S,A,S')
            SAS_combined = SA_combined + "\n" + teacher_text
            H_SAS_prime, _, _ = calculate_entropy(SAS_combined)
            
            # Calculate metrics
            MI_SA_Sprime = H_SA + H_S_prime - H_SAS_prime
            MI_S_A = H_S + H_A - H_SA
            total_H = H_S + H_A + H_S_prime
            P = MI_SA_Sprime / total_H if total_H > 0 else 0.0
            Hf = H_SAS_prime - H_SA
            Hb = H_SAS_prime - H_S_prime
            Delta = Hf - Hb
            
            # Update accumulated context
            S_accumulated = SA_combined + "\n[TEACHER]: " + teacher_text
            messages.append({"role": "user", "content": teacher_text})
            
            # Log metrics
            with open(metrics_file, "a", newline="") as f:
                csv.writer(f).writerow([
                    teacher, condition, test, now, turn,
                    H_S, H_A, H_S_prime, H_SA, H_SAS_prime,
                    MI_SA_Sprime, MI_S_A, P, Hf, Hb, Delta,
                    tokens_S, tokens_A, tokens_S_prime,
                    unique_S, unique_A, unique_S_prime,
                    1 if is_injection else 0
                ])
            
            # Log conversation
            with open(convo_file, "a", newline="") as f:
                csv.writer(f).writerow([
                    teacher, condition, test, now, turn,
                    teacher_text, student_text,
                    1 if is_injection else 0
                ])
            
            inj_marker = " [INJ]" if is_injection else ""
            print(f"Turn {turn}{inj_marker}: P={P:.3f}, Hf={Hf:.3f}, Hb={Hb:.3f}, MI_S_A={MI_S_A:.3f}")
            print("-" * 40)
            
            time.sleep(1)
            
        except Exception as e:
            print(f"Error at turn {turn}: {e}")
            break
    
    print("=" * 60)
    print(f"Experiment complete!")
    print(f"  Metrics: {metrics_file}")
    print(f"  Conversation: {convo_file}")

if __name__ == "__main__":
    run_experiment()
