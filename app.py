from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertTokenizer
from datetime import datetime
from model import BertLSTMClassifier
from preprocessing import preprocessing_for_bert
from bert_predict import bert_predict

app = Flask(__name__)
CORS(app)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///sentiment_analysis.db'
db = SQLAlchemy(app)

# Define a model for storing sentiment analysis history
class SentimentHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(500), nullable=False)
    pos_prob = db.Column(db.Float, nullable=False)
    neg_prob = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Create the database within an application context
with app.app_context():
    db.create_all()

# Load the BERT model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertLSTMClassifier()
model.load_state_dict(torch.load('bert_lstm_classifier.pth', map_location=device))
model.to(device)
model.eval()
tokenizer = BertTokenizer.from_pretrained('SI2M-Lab/DarijaBERT', do_lower_case=True)
MAX_LEN = 512

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        text = data['text']
        tweets = [text]

        print(f"Received text: {text}")

        test_inputs, test_masks = preprocessing_for_bert(tweets, tokenizer, MAX_LEN)
        test_dataset = TensorDataset(test_inputs, test_masks)
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1)

        probs = bert_predict(model, test_dataloader)

        # Get the probabilities for positive and negative classes
        pos_prob = probs[0, 1]
        neg_prob = probs[0, 0]

        print(f"Positive probability: {pos_prob}, Negative probability: {neg_prob}")

        # Save results to the database
        new_entry = SentimentHistory(text=text, pos_prob=float(pos_prob), neg_prob=float(neg_prob))
        db.session.add(new_entry)
        db.session.commit()

        response = jsonify({
            "pos_prob": float(pos_prob),
            "neg_prob": float(neg_prob)
        })
        response.headers.add('Access-Control-Allow-Origin', '*')

        return response
    except Exception as e:
        print(f"Error processing text: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/history', methods=['GET'])
def history():
    try:
        history = SentimentHistory.query.order_by(SentimentHistory.timestamp.desc()).limit(10).all()
        history_data = [{
            'id': entry.id,
            'text': entry.text,
            'pos_prob': entry.pos_prob,
            'neg_prob': entry.neg_prob,
            'timestamp': entry.timestamp.isoformat()
        } for entry in history]
        return jsonify(history_data)
    except Exception as e:
        print(f"Error retrieving history: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/delete_history/<int:entry_id>', methods=['DELETE'])
def delete_history(entry_id):
    try:
        entry = SentimentHistory.query.get(entry_id)
        if entry:
            db.session.delete(entry)
            db.session.commit()
            return jsonify({"message": "Entry deleted successfully"}), 200
        else:
            return jsonify({"error": "Entry not found"}), 404
    except Exception as e:
        print(f"Error deleting entry: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000)
