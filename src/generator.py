import os
import openai
from openai_client import CHAT_ENGINE

def generate_response(user_query: str, context_snippets: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful support assistant."},
        {"role": "user",   "content": f"Context:\n{context_snippets}\n\nQuestion: {user_query}"}
    ]
    resp = openai.ChatCompletion.create(
        engine=CHAT_ENGINE,
        messages=messages,
        temperature=0.7,
        max_tokens=256,
        top_p=0.6,
        frequency_penalty=0.7
    )   
    return resp.choices[0].message.content

if __name__ == "__main__":
    sample = generate_response(
        user_query="How do I reset my password?",
        context_snippets="To reset your password, go to Settings > Security. If you forget it, click 'Forgot Password'."
    )
    print(sample)
