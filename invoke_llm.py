import requests
from dotenv import load_dotenv
import os
import json

load_dotenv()
def invoke_gemini(query, context):
    prompt = f"""
    You are a helpful faq assistant. Answer the user's question based on the provided context.
    context: {context}
    user's question: {query}
    """
    try:

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={os.getenv('GEMINI_API_KEY')}"
        headers = {
            "Content-Type": "application/json",
        }
        data = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }]
        }

        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        response_json = response.json()

        generated_text = response_json["candidates"][0]["content"]["parts"][0]["text"]
        generated_text = generated_text.replace("```json", "").replace("```", "").strip()
        return generated_text
    except requests.exceptions.RequestException as e:
        print(f"Error invoking Gemini API: {e}")
        return None


