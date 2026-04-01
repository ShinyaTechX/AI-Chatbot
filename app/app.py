from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)

MODEL_PATH = "../model/gpt2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

# Fix padding token issue
tokenizer.pad_token = tokenizer.eos_token

# Store conversation
conversation = "This is a conversation between a human and a helpful assistant.\n"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    global conversation

    user_input = request.json["message"]
    conversation += f"Human: {user_input}\nAssistant: "

    # Limit memory (VERY IMPORTANT)
    conversation = conversation[-1000:]

    # Tokenize input
    inputs = tokenizer.encode(conversation, return_tensors="pt")

    # Generate response
    outputs = model.generate(
        inputs,
        max_length=inputs.shape[1] + 80,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract ONLY new reply
    reply = generated_text[len(conversation):]
    reply = reply.split("Human:")[0].strip()
    conversation += reply + "\n"
    print("BOT:", reply)
    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(debug=False)