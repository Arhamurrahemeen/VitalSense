# Skill: Groq API Integration for VitalSense

## Description
Replaces Gemini with Groq's llama-3.3-70b-versatile for real-time LLM health feedback due to higher rate limits and lower latency.

## Context
- Files: `ai_feedback.py`
- When: LLM health feedback generation in live webcam stream

## API Pattern
```python
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ],
    max_tokens=100,
    temperature=0.7
)
feedback = response.choices[0].message.content