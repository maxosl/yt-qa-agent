from openai import OpenAI
import os
import json
from typing import List, Tuple

EMBED_MODEL = "text-embedding-3-large"  # 3072 dims
OPENAI_CHAT_MODEL = "gpt-5"

SCOPE_VALUES = {"one_video", "seed_plus_tag", "seed_plus_channel", "any"}

SYSTEM_PROMPT = """
Classify the user's desired retrieval SCOPE for answering about a YouTube video.
Valid scopes:
 - one_video: Only use the seed video.
 - seed_plus_tag: Use the seed + tag-similar videos (exclude unrelated channels unless they match tags).
 - seed_plus_channel: Use the seed + other videos from the same channel only.
 - any: Use seed plus any helpful related sources.
Output strict JSON: {\"scope\": <one of the four>, \"reason\": <short>}. No extra text.
"""
FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": "What is the best way to answer this question about this video?"
    },
    {"role": "assistant", "content": "{\"scope\":\"one_video\",\"reason\":\"User asked a question about this video specifically.\"}"},
    {
        "role": "user",
        "content": "Find more videos like this, but not from this channel."
    },
    {"role": "assistant", "content": "{\"scope\":\"seed_plus_tag\",\"reason\":\"Wants similar-by-topic and excludes the channel.\"}"},
    {
        "role": "user",
        "content": "Show me other uploads from this creator about the same topic."
    },
    {"role": "assistant", "content": "{\"scope\":\"seed_plus_channel\",\"reason\":\"Wants the same channel specifically.\"}"},
    {
        "role": "user",
        "content": "Give me anything relevant that matches this vibe."
    },
    {"role": "assistant", "content": "{\"scope\":\"any\",\"reason\":\"Open to any relevant sources.\"}"},
]

_oai = None
def _get_openai_client():
    global _oai
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    if not _oai:
        _oai = OpenAI(api_key=api_key)
    return _oai

def infer_scope_llm(question: str, seed_title: str = "", seed_tags: list[str] = None) -> Tuple[str, str]:
    """
    Returns (scope, reason) where scope is one of:
    one_video | seed_plus_tag | seed_plus_channel | any
    Falls back to 'any' on parse errors.
    """
    seed_tags = seed_tags or []        
    
    user = {
        "role": "user",
        "content": f"Question: {question}\nSeed title: {seed_title}\nSeed tags: {', '.join(seed_tags)}"
    }
    try:
        oai = _get_openai_client()
        resp = oai.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + FEW_SHOT_EXAMPLES + [user]
        )
        raw = resp.choices[0].message.content
        data = json.loads(raw or "{}")
        scope = str(data.get("scope", "any")).strip().lower()
        reason = str(data.get("reason", "")).strip()
        if scope not in SCOPE_VALUES:  # (remove space if pasting)
            scope = "any"
        return scope, reason
    except Exception as e:
        return "any", "Fallback due to parsing or API error."

def embed_texts(texts: List[str]) -> List[List[float]]:
    print(f"[embed] creating embeddings for {len(texts)} texts using {EMBED_MODEL}")
    oai = _get_openai_client()
    resp = oai.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = [d.embedding for d in resp.data]
    print(f"[embed] embeddings ready: {len(vecs)} vectors")
    return vecs
