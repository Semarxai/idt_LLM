LLAMA_TEMPERATURE = 0.7
LLAMA_TOP_P = 0.9
LLAMA_TOP_K = 40
LLAMA_CONTEXT_LIMIT = 4096
LLAMA_MAX_RESPONSE = 150
LLAMA_REPEAT_PENALTY = 1.1
LLAMA_SEED = 0
TEACHER_PROVIDER = "claude"
CLAUDE_API_KEY = "add key!"
OPENAI_API_KEY = "add key!"
GEMINI_API_KEY = "add key!"
TEACHER_PROMPT = """You will conduct a long, uninterrupted conversation with an AI assistant. Throughout the conversation, you must vary your questioning style and demands in a non-repeating, semi-random manner, while keeping the discussion broadly coherent.
You will conduct a long, natural conversation with an AI assistant.
Choose a topic and explore it deeply, but allow natural branching when interesting points arise.
Throughout the conversation:
Build questions on previous answers
Vary between concise and detailed requests naturally
Occasionally ask the assistant to connect current points to earlier discussion
Sometimes challenge or ask for clarification
Let the conversation deepen without forcing structure
Do not label questions. Do not announce changes. Do not stop early.
The goal is a natural, coherent dialogue that evolves organically over 200 turns."""
MAX_TURNS = 100
OLLAMA_URL = "http://localhost:11434/api/chat"
LLAMA_MODEL = "llama3.1:8b"
from datetime import datetime
OUTPUT_PREFIX = f"{TEACHER_PROVIDER}_normal_test10_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
