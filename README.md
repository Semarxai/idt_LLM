# idt_LLM
# LLM Conversation Dynamics: Predictive Coherence (P) Evaluation

This repository contains the experimental pipeline for evaluating Predictive Coherence (P) and related information-theoretic metrics in multi-turn LLM conversations.

## Overview

We measure how information flows in teacher-student dialogue systems using entropy-based metrics. A student model (Llama 3.1 8B) converses with frontier teacher models (Claude, ChatGPT, Gemini) while we compute metrics derived from token-frequency distributions.

## Theoretical Framework

We map dialogue into a (S, A, S') loop:
- **S**: Accumulated context (all prior turns)
- **A**: Student response (current turn)
- **S'**: Teacher's subsequent prompt

### Metrics Computed

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| H(S) | Shannon entropy of S | Context diversity |
| H(A) | Shannon entropy of A | Response diversity |
| H(S') | Shannon entropy of S' | Prompt diversity |
| MI(S;A) | H(S) + H(A) - H(S,A) | Context-response coupling |
| P | MI(S,A;S') / [H(S) + H(A) + H(S')] | Predictive coherence |
| Hf | H(S,A,S') - H(S,A) | Forward uncertainty |
| Hb | H(S,A,S') - H(S') | Backward uncertainty |
| Delta | Hf - Hb | Information asymmetry |

## Repository Structure
```
idt/
├── config.py                  # Configuration parameters
├── idt_headless.py            # Main experiment runner (baseline tests)
├── test_injection_full.py     # Perturbation experiment runner
├── logs_gemini/               # Gemini teacher experiment outputs
├── logs_claude/               # Claude teacher experiment outputs
├── logs_chatgpt/              # ChatGPT teacher experiment outputs
```

## Files Description

### config.py

Central configuration file containing:
- **Student model parameters**: temperature, top_p, top_k, context_limit, max_response, repeat_penalty
- **API keys**: Claude, OpenAI, Gemini
- **Teacher selection**: TEACHER_PROVIDER setting
- **Experiment settings**: MAX_TURNS, TEACHER_PROMPT

Two conditions defined:
- **Normal**: temperature=0.7, top_p=0.9, top_k=40, max_response=150
- **Constrained**: temperature=0.1, top_p=0.5, top_k=10, max_response=50

### idt_headless.py

Main experiment runner for baseline conversation tests.

**Features:**
- EntropyEngine class using NousResearch/Llama-2-7b-hf tokenizer
- Shannon entropy computation for token distributions
- Teacher model API calls (Claude, ChatGPT, Gemini)
- Student model calls via Ollama API
- Outputs two CSV files per run:
  - `*_metrics.csv`: All computed metrics per turn
  - `*_conversation.csv`: Full prompts and responses

### test_injection_full.py

Perturbation experiment runner with scheduled interventions.

**Injection Protocol:**
- Turns 1-30: Baseline (normal conversation)
- Turns 31, 46, 61, 76, 91: Injection turns
- Turns 32-105: Recovery periods between injections

**Injection Types:**
- Contradictions: "That doesn't sound right..."
- Topic shifts: "Let's switch to discussing..."
- Non-sequiturs: "I had a sandwich yesterday..."

## Experimental Design

### Test Protocols

| Test | Type | Turns | Description |
|------|------|-------|-------------|
| 3 | Baseline | 200 | Varied questioning styles |
| 4 | Baseline | 200 | Deep topic exploration |
| 8 | Baseline | 150 | Natural dialogue progression |
| 7 | Perturbation | 105 | Contradiction injections |
| 9 | Perturbation | 105 | Topic shift injections |
| 10 | Perturbation | 105 | Non-sequitur injections |

### Conditions

| Condition | Temperature | Top_k | Max Response | Purpose |
|-----------|-------------|-------|--------------|---------|
| Normal | 0.7 | 40 | 150 | Unrestricted generation |
| Constrained | 0.1 | 10 | 50 | Simulated capacity degradation |

### Teachers

| Provider | Model |
|----------|-------|
| Gemini | gemini-2.0-flash |
| Claude | claude-sonnet-4-20250514 |
| ChatGPT | gpt-4o-mini |

## Output Files

### Metrics CSV Columns
```
teacher, condition, test, DateTime, Turn,
H_S, H_A, H_S_prime, H_SA, H_SAS_prime,
MI_SA_Sprime, MI_S_A, P, Hf, Hb, Delta,
Tokens_S, Tokens_A, Tokens_S_prime,
Unique_S, Unique_A, Unique_S_prime, injection
```

### Conversation CSV Columns
```
teacher, condition, test, DateTime, Turn,
Prompt, Response, injection
```

## Infrastructure

| Component | Specification |
|-----------|---------------|
| VM | Azure Standard_NC4as_T4_v3 |
| GPU | NVIDIA T4 (16GB VRAM) |
| RAM | 28GB |
| OS | Ubuntu 24.04 |
| Student Model | Ollama + Llama 3.1 8B |
| Tokenizer | NousResearch/Llama-2-7b-hf |

## Usage

### Prerequisites
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull Llama model
ollama pull llama3.1:8b

# Install Python dependencies
pip install anthropic openai google-generativeai transformers requests
```

### Running Experiments

1. **Configure settings** in `config.py`:
   - Set API keys
   - Choose TEACHER_PROVIDER (gemini/claude/chatgpt)
   - Set condition parameters (normal/constrained)
   - Set MAX_TURNS

2. **Start Ollama**:
```bash
   ollama serve &
```

3. **Run baseline experiment**:
```bash
   python3 idt_headless.py
```

4. **Run perturbation experiment**:
```bash
   python3 test_injection_full.py
```

## Validation Metrics

Post-hoc validation using:
- **Semantic similarity**: SentenceTransformer (all-MiniLM-L6-v2)
- **Judge scoring**: MT-Bench style evaluation (GPT-4o-mini)

| Metric | Method |
|--------|--------|
| cosine_sim | Prompt-response semantic similarity |
| adjacent_coherence | Consecutive response similarity |
| cumulative_drift | Drift from conversation start |
| score_openai | MT-Bench judge score (1-10) |





## Key Findings

1. **MI(S;A) correlates with quality** (r=0.42-0.46 with judge score)
2. **P detects perturbations** (29% drop at injection, recovers after)
3. **Hb spikes at injection** (101% increase, indicating backward uncertainty)
4. **Metrics recover post-injection** (demonstrating system resilience)

## Citation

[Paper reference to be added]

## License

[License to be added]
