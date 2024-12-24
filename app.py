from flask import Flask, request, jsonify
from transformers import pipeline

# Load the Llama model
llama_pipeline = pipeline("text-generation", model="meta-llama/Llama-2-13b-hf")

app = Flask(__name__)

@app.route("/api/completions", methods=["POST"])
def generate_completion():
    data = request.json
    prompt = data.get("prompt", "")
    max_tokens = data.get("max_tokens", 50)
    temperature = data.get("temperature", 0.7)

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    response = llama_pipeline(prompt, max_length=max_tokens, temperature=temperature)
    return jsonify({"completions": [response[0]["generated_text"]]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
