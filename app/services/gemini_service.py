
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

client = None

if API_KEY:
    client = genai.Client(api_key=API_KEY)


def generate_summary(text: str) -> str:
    if client is None:
        raise RuntimeError("Gemini API key is missing. Check your .env file.")

    prompt = f"""
    You are an expert text summarizer.

    Summarize the following text clearly and accurately.
    Preserve the important facts and key points.
    Avoid adding information that is not in the original text.

    Text:
    {text}
    """

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty summary.")

    return response.text.strip()
