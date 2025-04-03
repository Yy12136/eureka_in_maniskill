import openai
import os

def test_connection():
    print(f"Current API base: {openai.api_base}")
    print(f"API key: {openai.api_key[:8]}..." if openai.api_key else "Not set")
    
    try:
        response = openai.ChatCompletion.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7
        )
        print("Success!")
        print("Response:", response.choices[0].message.content)
    except Exception as e:
        print("Error:", str(e))

if __name__ == "__main__":
    test_connection()