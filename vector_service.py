
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModel
from flask_wtf.csrf import CSRFProtect
import torch
import logging

app = Flask(__name__)
csrf = CSRFProtect(app)
CORS(app, resources={r"/v1/embeddings": {"origins": "http://localhost:3000"}}, supports_credentials=True)

# Configure logging
logging.basicConfig(filename='file.log', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
app.logger.setLevel(logging.DEBUG)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
model = AutoModel.from_pretrained("bert-base-multilingual-cased")

@csrf.exempt  # Disable CSRF protection for this route
@app.route('/v1/embeddings', methods=['POST'])
def vectorize_text():
    try:
        data = request.get_json()
        text_input = data.get('input', '')
        # model_name = data.get('model', 'bert-base-multilingual-cased') # Model is fixed for now
        # dimensions = data.get('dimensions') # Dimensions not used by current model loading

        if not text_input:
            app.logger.error("Error in /v1/embeddings: 'input' field is missing or empty")
            return jsonify({"error": "'input' field is required"}), 400

        inputs = tokenizer(text_input, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        # Calculate the mean of the last hidden state tokens to get a sentence vector
        vector = outputs.last_hidden_state.mean(dim=1).squeeze().numpy().tolist()
        
        # Ensure vector is a list of numbers (floats or ints)
        if not isinstance(vector, list) or not all(isinstance(n, (int, float)) for n in vector):
             # If squeeze results in a 0-dim tensor (single number), tolist() might return a scalar
            if isinstance(vector, (int, float)):
                vector = [vector] # Wrap it in a list
            else: # If it's some other non-list structure or contains non-numbers, log error
                app.logger.error(f"Error in /v1/embeddings: Vector is not a list of numbers. Got: {type(vector)}")
                return jsonify({"error": "Failed to generate a valid embedding vector format"}), 500

        response_data = {"data": [{"embedding": vector}]}
        return jsonify(response_data)
    except Exception as e:
        app.logger.error(f"Error in /v1/embeddings route: {str(e)}")
        return jsonify({"error": str(e)}), 500

@csrf.exempt  # Disable CSRF protection for this route
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"})


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5050)
