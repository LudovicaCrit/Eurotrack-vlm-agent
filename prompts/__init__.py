"""
Prompt loader per VLM Agent
"""
import os

PROMPTS_DIR = os.path.dirname(__file__)

def load_prompt(filename):
    """Carica un prompt da file .md"""
    path = os.path.join(PROMPTS_DIR, filename)
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip()

# Carica i prompt

SYSTEM_PROMPT = load_prompt('system.md')
DRIVING_ANALYSIS_PROMPT = load_prompt('driving.md')