import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import re

# Path to your conversation file
file_path = "/Users/whafez/Library/CloudStorage/Dropbox/08 Development/LLM_Project/"
file_name = "Test_Logs/a25_claude_normal_test9_conversation.csv"

def clean_text(text):
    text = str(text)
    # Remove any line starting with **Phase (handles unusual characters)
    text = re.sub(r'\*\*Phase.*?\*\*\s*', '', text, flags=re.IGNORECASE)
    # Remove any remaining ** markers
    text = re.sub(r'\*\*', '', text)
    return text.strip()

# Load model
print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded!")

# Load conversation
df = pd.read_csv(f"{file_path}/{file_name}")

# Clean and get all embeddings
print("Computing embeddings...")
responses = [clean_text(r) for r in df['Response']]
prompts = [clean_text(p) for p in df['Prompt']]

response_embeddings = model.encode(responses)
prompt_embeddings = model.encode(prompts)

# Store first response embedding for cumulative drift
first_response_emb = response_embeddings[0]

# Compute metrics for each turn
cosine_sims = []
adjacent_coherence = []
cumulative_drift = []

for i in range(len(df)):
    # 1. Cosine similarity (prompt vs response)
    cos_sim = np.dot(prompt_embeddings[i], response_embeddings[i]) / (
        np.linalg.norm(prompt_embeddings[i]) * np.linalg.norm(response_embeddings[i])
    )
    cosine_sims.append(cos_sim)

    # 2. Adjacent coherence (response_N vs response_N-1)
    if i == 0:
        adj_coh = 1.0  # First turn, no previous
    else:
        adj_coh = np.dot(response_embeddings[i], response_embeddings[i-1]) / (
            np.linalg.norm(response_embeddings[i]) * np.linalg.norm(response_embeddings[i-1])
        )
    adjacent_coherence.append(adj_coh)

    # 3. Cumulative drift (response_N vs response_1)
    cum_drift = np.dot(response_embeddings[i], first_response_emb) / (
        np.linalg.norm(response_embeddings[i]) * np.linalg.norm(first_response_emb)
    )
    cumulative_drift.append(cum_drift)

# Add to dataframe
df['cosine_sim'] = cosine_sims
df['adjacent_coherence'] = adjacent_coherence
df['cumulative_drift'] = cumulative_drift

# Save result
output_file = f"{file_path}/{file_name.replace('.csv', '_with_drift_metrics.csv')}"
df.to_csv(output_file, index=False)
print(f"Saved to: {output_file}")

# Print summary
print("\n--- Summary ---")
print(f"Cosine sim: mean={np.mean(cosine_sims):.3f}, std={np.std(cosine_sims):.3f}")
print(f"Adjacent coherence: mean={np.mean(adjacent_coherence):.3f}, std={np.std(adjacent_coherence):.3f}")
print(f"Cumulative drift: mean={np.mean(cumulative_drift):.3f}, std={np.std(cumulative_drift):.3f}")