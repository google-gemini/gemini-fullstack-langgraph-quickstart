import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ Error: GEMINI_API_KEY is not set in .env")
else:
    try:
        genai.configure(api_key=api_key)
        print("✅ API Key found. Listing models...")
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                print(f"Model: {m.name}")
    except Exception as e:
        print(f"❌ Error: {e}")
