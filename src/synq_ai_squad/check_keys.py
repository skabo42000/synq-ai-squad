"""Check which AI models your API keys can actually use.

Run:  uv run python -m synq_ai_squad.check_keys
It never prints the keys themselves, only whether they work.
"""

import os

from dotenv import load_dotenv

load_dotenv()  # reads .env into environment variables


def check_gemini() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        print("Gemini: no GOOGLE_API_KEY in .env - skipped")
        return
    from google import genai

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    try:
        names = [m.name.removeprefix("models/") for m in client.models.list()]
    except Exception as e:
        print(f"Gemini: key did not work ({type(e).__name__}: {e})")
        return
    print(f"Gemini: key works, {len(names)} models listed. Gemini chat models:")
    for n in sorted(n for n in names if n.startswith("gemini")):
        print("   -", n)


def check_groq() -> None:
    if not os.getenv("GROQ_API_KEY"):
        print("Groq: no GROQ_API_KEY in .env - skipped")
        return
    from groq import Groq

    try:
        models = Groq().models.list().data
    except Exception as e:
        print(f"Groq: key did not work ({type(e).__name__}: {e})")
        return
    print(f"Groq: key works, {len(models)} models:")
    for m in sorted(models, key=lambda m: m.id):
        print("   -", m.id)


if __name__ == "__main__":
    check_gemini()
    print()
    check_groq()
