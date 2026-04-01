# pipeline is a high-level API from Hugging Face. It lets you use powerful models in 1 line. Loads Hugging Face pipeline system. Simplifies using large models like GPT-2

from transformers import pipeline

# Create chatbot
chatbot = pipeline("text-generation", model="gpt2")   # "text-generation" -> task type. "gpt2" -> loads pretrained GPT-2 model. GPT-2 predicts the next words based on input text. Behind the scenes: Downloads model(first time only). Loads tokenizer. Loads neural network(~100M+ parameters). GPT-2 was trained on massive internet text

conversation = ""   # Initialize conversaation memory. This variable stores entire chat history. Starts empty

print("GPT Chatbot Ready! (type 'quit')")   # Print ready message. Just UI message for user

while True:   # Start infinite loop. Keeps chatbot running forever. Stops only when user types "quit"
    user_input = input("You: ")   # Get user input

    if user_input.lower() == "quit":   
        break

    conversation += f"User: {user_input}\nBot:"   # Add user message to conversation. This format is very important-it teaches GPT: "User" -> human message, "Bot:" -> model should respond

    response = chatbot(conversation, max_length=200, pad_token_id=50256)   # Generate response. GPT-2 receives full conversation: (User: Hello, Bot: ). It continues the text (User: Hello, Bot: Hi there! How can I help you?). max_length=200: Maximum total text lenth(input+output). pad_token_id=50256: GPT-2 has no default padding token, 50256 = end-of-text token. Prevents warnings/errors

    reply = response[0]['generated_text'].split("Bot:")[-1]   # Extract only the reply. Why needed? GPT-2 returns FULL text: (User:Hello, Bot:Hi there! How can I help you?). Step-by-step: (1. Split text at "Bot:", 2. Take last part). Result: Hi there! How can I help you? 

    print("Bot:", reply.strip())   # Print response. strip() removes extra spaces/newlines. 

    conversation += reply + "\n"   # Update conversation memory