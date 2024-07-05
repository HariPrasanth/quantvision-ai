import openai
import os

from dotenv import load_dotenv

load_dotenv()
# Set your OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Define the messages for the chat model
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me a joke"}
]

# Call the OpenAI API to get a response
response = openai.ChatCompletion.create(
  model="gpt-4o",
  messages=messages,
  max_tokens=50
)

# Print the response
print(response.choices[0].message['content'].strip())
