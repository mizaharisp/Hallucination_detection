
import os
from google import genai
from dotenv import load_dotenv
from google.genai import errors

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from the .env file.")

client = genai.Client(api_key=api_key)


def generate_response(prompt):
    try:
        response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
)

        if not response.text:
            raise RuntimeError("Gemini returned an empty response.")

        return response.text

    except errors.APIError as e:
        print(f"Gemini API error: {e.code} - {e.message}")
        raise