import requests
import os

def construct_persona_from_intro(intro_text, llm_api_url, llm_api_key):
    """
    Use the LLM to generate a persona description from the intro text.
    Returns a concise persona prompt string.
    """
    system_prompt = (
        "You are an expert at extracting personas and tone from background information. "
        "Given the following introduction, write a concise persona description (1-2 sentences) that can be used as a system prompt for a chatbot to answer as this person. "
        "Focus on style, tone, and relevant background."
    )
    data = {
        "model": "meta-llama/llama-guard-4-12b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": intro_text}
        ]
    }
    headers = {"Authorization": f"Bearer {llm_api_key}", "Content-Type": "application/json"}
    response = requests.post(llm_api_url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"[Persona construction error: {response.status_code}]"
