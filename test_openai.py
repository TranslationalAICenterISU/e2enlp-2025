from openai import OpenAI

client = OpenAI(
  api_key="XYZ"
)

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": """You are a helpful assistant for Manufacturing and Operations Management.
     Please answer the following question: What is the difference between a product and a service?"""},
  ]
)

print(completion.choices[0].message);
