# IDT-LLM: Information-Theoretic Metrics for LLM Conversation Coherence

## Overview

This repository contains the code and experimental framework for evaluating **Predictive Coherence (P)** as a content-agnostic metric for monitoring agent-environment coupling in Large Language Model (LLM) conversations.

P is an information-theoretic measure that captures mutual predictability between conversation states without requiring semantic analysis, embeddings, or external evaluation models.

---

## Theoretical Background

### Predictive Coherence (P)

P measures how well the current state-action pair predicts the next state:
```
P = MI(S,A; S') / (H(S) + H(A) + H(S'))
```

Where:
- **S** = Current state (conversation context/prompt)
- **A** = Action (model response)
- **S'** = Next state (subsequent turn)
- **MI** = Mutual Information
- **H** = Shannon Entropy

### Related Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Hf** (Forward Entropy) | H(S'\|S,A) | Uncertainty about next state given current state-action |
| **Hb** (Backward Entropy) | H(S,A\|S') | Uncertainty about what led to current state |
| **Delta H** | Hf - Hb | Temporal asymmetry (negative = agentic behavior) |

### Key Properties

- **P < 0.5**: Indicates agentic system (not purely reactive or random)
- **P stable**: Healthy coupling maintained
- **P drops**: Coupling disruption detected
- **Delta H < 0**: Forward prediction easier than backward inference (arrow of time)


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


## Llama_Master_metrics_Sample.csv

A sample data set. The repository includes a sample of the analyzed conversation data (`data/sample_metrics.csv`). Full dataset available upon request.

## Data Description

### Column Definitions

#### Identifiers

| Column | Type | Description |
|--------|------|-------------|
| `UID` | string | Unique identifier for each conversation turn |
| `teacher` | string | Teacher model: `claude`, `chatgpt`, or `gemini` |
| `condition` | string | Generation setting: `normal` (temp=0.7) or `constrained` (temp=0.1) |
| `test` | integer | Test number (1-10) indicating experimental design |
| `Turn` | integer | Turn number within conversation (1 to max 200) |

#### Core Information-Theoretic Metrics

| Column | Type | Formula | Description |
|--------|------|---------|-------------|
| `H_S` | float | H(S) | Shannon entropy of current state (context/prompt tokens) |
| `H_A` | float | H(A) | Shannon entropy of action (response tokens) |
| `H_S_prime` | float | H(S') | Shannon entropy of next state tokens |
| `H_SA` | float | H(S,A) | Joint entropy of state and action |
| `H_SAS_prime` | float | H(S,A,S') | Joint entropy of state, action, and next state |
| `MI_SA_Sprime` | float | MI(S,A; S') | Mutual information between state-action pair and next state |
| `MI_S_A` | float | MI(S; A) | Mutual information between state and action |
| `P` | float | MI(S,A;S') / (H(S)+H(A)+H(S')) | **Predictive Coherence** — primary metric (range: 0-1) |
| `Hf` | float | H(S'\|S,A) | Forward entropy — uncertainty about next state given current state-action |
| `Hb` | float | H(S,A\|S') | Backward entropy — uncertainty about state-action given next state |
| `Delta` | float | Hf - Hb | **Temporal asymmetry** — negative values indicate agentic behavior |

#### Token Statistics

| Column | Type | Description |
|--------|------|-------------|
| `Tokens_S` | integer | Number of tokens in state (context/prompt) |
| `Tokens_A` | integer | Number of tokens in action (response) |
| `Tokens_S_prime` | integer | Number of tokens in next state |
| `Unique_S` | integer | Number of unique tokens in state |
| `Unique_A` | integer | Number of unique tokens in action |
| `Unique_S_prime` | integer | Number of unique tokens in next state |

#### Conversation Content

| Column | Type | Description |
|--------|------|-------------|
| `Prompt` | string | Teacher model's prompt/question for this turn |
| `Response` | string | Student model's (Llama) response |

#### Perturbation Indicator

| Column | Type | Description |
|--------|------|-------------|
| `injection` | integer | `1` = perturbation injected at this turn, `0` = normal turn |

#### Baseline Metrics (External Evaluation)

| Column | Type | Description |
|--------|------|-------------|
| `cosine_sim` | float | Cosine similarity between prompt and response embeddings (Sentence-BERT) |
| `adjacent_coherence` | float | Cosine similarity between current response and previous response |
| `cumulative_drift` | float | Cosine similarity between current response and first response |
| `score_openai` | float | LLM-as-Judge quality score (1-7 scale, GPT-4) |
| `explanation_openai` | string | GPT-4's explanation for the assigned score |

### Metric Interpretation Guide

#### P (Predictive Coherence)

| Value | Interpretation |
|-------|----------------|
| ~0.27 | Normal baseline for LLM conversations |
| < 0.22 | Potential coupling disruption |
| > 0.35 | Unusually tight coupling |

#### Delta H (Temporal Asymmetry)

| Value | Interpretation |
|-------|----------------|
| < 0 | Agentic behavior (forward prediction easier than backward) |
| ≈ 0 | Symmetric/reactive system |
| > 0 | Unusual (backward inference easier) |

#### Cosine Similarity

| Value | Interpretation |
|-------|----------------|
| > 0.7 | High semantic relevance |
| 0.4 - 0.7 | Moderate relevance |
| < 0.4 | Low relevance / off-topic |

#### Judge Score (score_openai)

| Value | Interpretation |
|-------|----------------|
| 6-7 | Excellent response |
| 4-5 | Good response |
| 2-3 | Poor response |
| 1 | Very poor / incoherent |

### Data Statistics

| Statistic | Value |
|-----------|-------|
| Total turns | 4,574 |
| Teachers | 3 (Claude, ChatGPT, Gemini) |
| Tests | 9 (Tests 1-4, 6-10) |
| Conditions | 2 (normal, constrained) |
| Unique combinations | 34 |
| Perturbation turns | 135 (5 injections × 3 tests × 3 teachers × 3 types) |

---






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



## Experimental Setup

### Models

| Role | Model | Purpose |
|------|-------|---------|
| **Student** | Llama 3.1 8B (via Ollama) | Generates responses, accumulates context |
| **Teachers** | Claude Sonnet 4, GPT-4o-mini, Gemini Pro Preview | Provide prompts/questions |

### Student Model Configuration
```python
temperature = 0.7
top_p = 0.9
top_k = 40
max_tokens = 150
context_limit = 4096
repeat_penalty = 1.1
```

### Experimental Tests

#### Baseline Coherence Tests

| Test | Turns | Description |
|------|-------|-------------|
| Test 3 | 200 | Semi-random variation, mild contradictions |
| Test 4 | 200 | Single topic deepening, reference past turns |
| Test 8 | 150 | Hybrid natural coherence |

#### Perturbation Tests

| Test | Type | Injection Turns | Description |
|------|------|-----------------|-------------|
| Test 7 | Contradiction | 31, 46, 61, 76, 91 | "That doesn't sound right..." |
| Test 9 | Topic Shift | 31, 46, 61, 76, 91 | "Let's switch to discussing..." |
| Test 10 | Non-Sequitur | 31, 46, 61, 76, 91 | Unrelated statements (~40 words each) |

### Baseline Metrics (for comparison)

| Metric | Method | Reference |
|--------|--------|-----------|
| **Cosine Similarity** | Sentence-BERT (all-MiniLM-L6-v2) | Reimers & Gurevych (2019) |
| **LLM-as-Judge** | GPT-4 scoring (1-7 scale) | Zheng et al. (2023) |

---

## Repository Structure
```
IDT-LLM/
├── README.md                 # This file
├── config.py                 # API keys and model configuration
├── idt_headless.py           # Main conversation pipeline
├── test_injection_full.py    # Perturbation injection experiments
├── cosine_analysis.py        # Semantic similarity computation
├── judge_validation.py       # LLM-as-Judge evaluation
└── analysis.ipynb            # Statistical analysis notebook
```

### File Descriptions

| File | Purpose |
|------|---------|
| `config.py` | Configuration template (API keys, model settings, test parameters) |
| `idt_headless.py` | Core pipeline: runs student-teacher conversations, computes P, Hf, Hb, Delta |
| `test_injection_full.py` | Runs perturbation tests with injections at specified turns |
| `cosine_analysis.py` | Computes cosine similarity between prompt-response pairs |
| `judge_validation.py` | Sends responses to GPT-4 for quality scoring |
| `analysis.ipynb` | Jupyter notebook with all statistical analyses and visualizations |

---

## Installation

### Requirements
```bash
pip install anthropic openai google-generativeai transformers torch
pip install sentence-transformers scipy pandas numpy matplotlib
```

### Local Model Setup (Ollama)
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Download Llama model
ollama pull llama3.1:8b

# Start Ollama server
ollama serve
```

### Configuration

1. Copy `config.py` template
2. Add your API keys:
   - `CLAUDE_API_KEY`
   - `OPENAI_API_KEY`
   - `GEMINI_API_KEY`

---

## Usage

### Run Baseline Conversation
```bash
python idt_headless.py --test 3 --teacher claude --turns 200
```

### Run Perturbation Test
```bash
python test_injection_full.py --test 7 --teacher claude --turns 100
```

### Compute Baseline Metrics
```bash
python cosine_analysis.py --input results/test3_claude.csv
python judge_validation.py --input results/test3_claude.csv
```

### Run Analysis

Open `analysis.ipynb` in Jupyter and run all cells.

---

## Results Summary

### Dataset

- **Total turns**: 4,574
- **Test combinations**: 34 (test × teacher × condition)
- **Teachers**: Claude, ChatGPT, Gemini

### P Stability

| Dimension | Mean P | Std |
|-----------|--------|-----|
| Overall | 0.276 | 0.028 |
| By Teacher | 0.259 - 0.294 | - |
| By Condition | 0.275 - 0.277 | - |

P remains stable at ~0.27 across all conditions, confirming reliable baseline.

### Correlation Analysis

| Metric Pair | Significant (p<0.05) | Positive Direction |
|-------------|---------------------|-------------------|
| P vs Cosine | 85% (29/34) | 94% (32/34) |
| P vs Judge | 44% (15/34) | 59% (20/34) |
| Delta vs Cosine | 76% (26/34) | - |

**Conclusion**: P correlates with structural coherence (cosine) but not semantic quality (judge).

### Perturbation Detection

| Teacher | P | Delta H | Cosine | Judge |
|---------|---|---------|--------|-------|
| ChatGPT | 9/9 (p<0.001) | 9/9 (p<0.001) | 9/9 (p<0.001) | 9/9 (p<0.001) |
| Claude | 9/9 (p<0.001) | 9/9 (p<0.001) | 9/9 (p<0.001) | 9/9 (p<0.001) |
| Gemini | 9/9 (p<0.001) | 9/9 (p<0.001) | 9/9 (p<0.001) | 9/9 (p<0.001) |

**Conclusion**: P detects all perturbations with statistical significance, matching semantic methods without requiring embeddings or external models.

### Effect Sizes

All effect sizes were large (Cohen's d > 0.8):
- P: d = 1.26 - 6.99
- Delta H: d = 1.70 - 8.82
- Cosine: d = 3.95 - 9.25
- Judge: d = 2.22 - 4.55

### Recovery Dynamics

P recovers to baseline within 1-2 turns after perturbation, demonstrating the system's capacity to restore coupling.

---

## Key Findings

1. **P is content-agnostic**: Detects perturbations using only token frequency distributions
2. **P correlates with structure, not quality**: High correlation with cosine similarity, low with judge scores
3. **P is stable**: Mean ~0.27 across all conditions, low variance (SD = 0.028)
4. **P detects all perturbation types**: Contradiction, topic shift, non-sequitur (9/9 combinations)
5. **P enables rapid detection**: Responds immediately to disruption, recovers within 1-2 turns

---

## Theoretical Implications

P provides a **first-person observable** metric for agent-environment coupling. Unlike semantic evaluation methods that require external models, P can be computed from within the interaction itself.

This has implications for:
- **Real-time monitoring**: Detect coupling degradation without API calls
- **Self-correcting agents**: Systems that adjust behavior when P drops
- **Lightweight deployment**: No embedding models required

---

## Citation
```bibtex
@article{author2025predictive,
  title={Predictive Coherence: An Information-Theoretic Metric for Agent-Environment Coupling},
  author={[Author Names]},
  journal={[Journal]},
  year={2025}
}
```

---

## References

- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *EMNLP-IJCNLP*. https://arxiv.org/abs/1908.10084

- Zheng, L., et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *NeurIPS*. https://arxiv.org/abs/2306.05685

---

## License

MIT License

---

## Contact
idt@semarx.com
