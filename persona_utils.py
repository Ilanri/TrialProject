import requests
import os

def construct_persona_from_intro(intro_text, llm_api_url, llm_api_key):
    """
    Use the LLM to generate a persona description from the intro text.
    Returns a concise persona prompt string.
    """
    system_prompt = (
      "You are an expert at extracting personas and tone from background information. "
      "Given the following introduction, create a **concise system prompt** for a chatbot. "
      "The system prompt should guide the chatbot to respond in a way that reflects the persona, style, and tone of the individual described in the introduction. "
      "Focus on the following elements: tone, personality traits, communication style, and any relevant background. "
      "The output should only be the **system prompt** that can be used by the chatbot. "
      "Do not add any additional text or explanations. Just the system prompt."
    )

    data = {
        "model": "llama-3.3-70b-versatile",
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
