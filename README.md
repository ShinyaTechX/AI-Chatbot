# Transformer Chatbot (GPT-style)

A simple web-based chatbot built using a **Transformer model (GPT-2)** with a Flask backend and interactive frontend.
This project demonstrates how to run a **local language model** and serve it through a web API.

## Features

- Local GPT-2 model (no API required)
- Interactive chat interface
- Maintains conversation context
- Fast Flask backend
- Easy to customize and extend

## Tech Stack

* **Backend:** Python, Flask
* **Model:** Hugging Face Transformers (GPT-2)
* **Frontend:** HTML, CSS, JavaScript
* **Libraries:** PyTorch, Transformers

## How It Works

1. User enters a message in the browser
2. Frontend sends a POST request to `/chat`
3. Flask backend processes the input
4. GPT-2 generates a response
5. Response is returned as JSON
6. Frontend displays the reply

---

## Example

**User:** Hello
**Bot:** Hi! How can I help you today?

## Limitations

* GPT-2 is not optimized for conversation
* Responses may be:

  * repetitive
  * off-topic
  * inconsistent

## License

This project is open-source and available under the MIT License.

## Acknowledgements

* Hugging Face Transformers
* OpenAI GPT architecture
* Flask framework

## Run
Python app/app.py
