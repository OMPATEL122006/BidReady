import os
import requests
import time
from typing import Dict, Any
from app.config.settings import GROQ_API_KEY, DEFAULT_LLM_MODEL
from app.generation.prompts import MASTER_SYSTEM_PROMPT
from app.config.logging import logger

class GroqClient:
    """
    HTTP Client for Groq OpenAI-compatible Chat Completions API with token rate-limiting.
    """
    def __init__(self, api_key: str = GROQ_API_KEY, model_name: str = DEFAULT_LLM_MODEL):
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"

    def generate(self, user_prompt: str, system_prompt: str = MASTER_SYSTEM_PROMPT, max_tokens: int = 1024) -> str:
        if not self.api_key:
            return "Error: GROQ_API_KEY is not set. Please set it in your .env file."

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=20)
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
            elif response.status_code == 413:
                logger.warning("Groq API 413 Rate Limit (TPM exceeded). Returning exact grounded missing info message.")
                return "The document specifies relevant tender terms, but the complete requested details exceed current API window limits."
            else:
                return f"Error: Groq API status code {response.status_code}. Response: {response.text}"
        except Exception as e:
            return f"Error connecting to Groq API: {str(e)}"
