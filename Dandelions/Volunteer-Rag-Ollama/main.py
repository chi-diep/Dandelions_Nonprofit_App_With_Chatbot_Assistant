from ollama import Client
client = Client()

while True:
    question = input("\nAsk a question (or type 'exit' to quit): ")
    if question.lower() == 'exit':
        break
    response = client.chat(
        model="llama2",
        messages=[
            {"role": "user", "content": question}
        ]
    )
    print(response['message']['content'])
