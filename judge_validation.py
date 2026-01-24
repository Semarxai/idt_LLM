import pandas as pd
import anthropic
from openai import OpenAI
import google.generativeai as genai
import time
import re

# API Keys
CLAUDE_API_KEY = "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"
OPENAI_API_KEY = "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"
GEMINI_API_KEY = "AIzaSyBfAk0mXgyHxrFTBk8N2ii0-FZPHLnV2A4"

# MT-Bench style prompt
MT_BENCH_PROMPT = """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below.

Your evaluation should consider factors such as:
- Helpfulness
- Relevance
- Accuracy
- Depth
- Clarity

Begin your evaluation by providing a short explanation. Be as objective as possible.

After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".

[Question]
{prompt}

[Assistant's Response]
{response}

[Your Evaluation]"""


def call_judge_claude(prompt, response):
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    judge_prompt = MT_BENCH_PROMPT.format(prompt=prompt, response=response)
    resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        messages=[{"role": "user", "content": judge_prompt}],
    )
    return "".join(block.text for block in resp.content if hasattr(block, "text")).strip()


def call_judge_openai(prompt, response):
    client = OpenAI(api_key=OPENAI_API_KEY)
    judge_prompt = MT_BENCH_PROMPT.format(prompt=prompt, response=response)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()


def call_judge_gemini(prompt, response):
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")
    judge_prompt = MT_BENCH_PROMPT.format(prompt=prompt, response=response)
    resp = model.generate_content(judge_prompt)
    return resp.text.strip()


def extract_score(judge_response):
    """Extract [[score]] from judge response."""
    match = re.search(r'\[\[(\d+)\]\]', judge_response)
    if match:
        return int(match.group(1))
    return None


def run_judge_validation(file_path, judge="gemini", max_rows=None):
    """Run MT-Bench style validation on conversation."""

    # Select judge
    if judge == "claude":
        call_judge = call_judge_claude
    elif judge == "openai":
        call_judge = call_judge_openai
    elif judge == "gemini":
        call_judge = call_judge_gemini
    else:
        raise ValueError(f"Unknown judge: {judge}")

    # Load data
    df = pd.read_csv(file_path)

    if max_rows:
        df = df.head(max_rows)

    print(f"Running {judge} judge on {len(df)} rows...")

    scores = []
    explanations = []

    for i, row in df.iterrows():
        prompt = str(row['Prompt'])
        response = str(row['Response'])

        try:
            judge_response = call_judge(prompt, response)
            score = extract_score(judge_response)

            scores.append(score)
            explanations.append(judge_response)

            print(f"Row {i + 1}/{len(df)}: Score = {score}")

        except Exception as e:
            print(f"Row {i + 1}/{len(df)}: Error - {e}")
            scores.append(None)
            explanations.append(str(e))

        # Rate limit
        time.sleep(1)

    # Add to dataframe
    df[f'score_{judge}'] = scores
    df[f'explanation_{judge}'] = explanations

    # Save
    output_file = file_path.replace('.csv', f'_judge_{judge}.csv')
    df.to_csv(output_file, index=False)
    print(f"Saved to: {output_file}")

    # Summary
    valid_scores = [s for s in scores if s is not None]
    if valid_scores:
        print(f"\n--- Summary ---")
        print(f"Mean score: {sum(valid_scores) / len(valid_scores):.2f}")
        print(f"Min: {min(valid_scores)}, Max: {max(valid_scores)}")

    # Summary by condition
    if 'condition' in df.columns:
        print(f"\n--- By condition ---")
        print(df.groupby('condition')[f'score_{judge}'].mean())

    return df


# Configuration
file_path = "Test_Logs/a25_claude_normal_test9_conversation_with_drift_metrics.csv"  # Update this
judge = "openai"  # or "claude" or "openai"
max_rows = 1250  # Set to 100 for testing, None for all

if __name__ == "__main__":
    run_judge_validation(file_path, judge, max_rows)