import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

print("Testing OpenRouter API connection...")
print("=" * 60)

# Initialize client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Test basic API call
response = client.chat.completions.create(
    model="nvidia/nemotron-3-super-120b-a12b:free",
    messages=[
        {
            "role": "user",
            "content": "Say 'API is working!' and nothing else."
        }
    ]
)

print(f"[SUCCESS] API Response:")
print(f"  {response.choices[0].message.content}")
print(f"\nModel used: {response.model}")
print(f"Tokens used: {response.usage.total_tokens}")