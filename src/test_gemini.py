"""Quick smoke-test: verifies Groq connectivity with the configured model."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from groq import Groq  # noqa: E402
from rag.llm_coach import GROQ_MODEL  # noqa: E402

client = Groq(api_key=os.environ["GROQ_API_KEY"])
prompt = "Hello"

response = client.chat.completions.create(
    model=GROQ_MODEL,
    messages=[
        {"role": "system", "content": "You are a warm, empathetic mental health wellness buddy. Keep responses short and comforting."},
        {"role": "user", "content": prompt},
    ],
)

full_response = response.choices[0].message.content
print(full_response)
