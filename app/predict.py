import os
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import pickle

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)

VOCAB_SIZE = 10000
MAX_LEN = 250
MODEL_PATH = os.path.join(parent_dir, 'models', 'distilbert_model_best.pth')
MODEL_NAME = 'distilbert-base-uncased'
# Load the saved model
model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()


# Load the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

def encode_texts(text_list):
    encoded_inputs = tokenizer(text_list, padding=True, truncation=True, max_length=MAX_LEN, return_tensors='pt')
    return encoded_inputs

def predict_sentiments(text_list):
    encoded_inputs = encode_texts(text_list)
    with torch.no_grad():
        outputs = model(**encoded_inputs)
        predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()

    sentiments = []
    for prediction in predictions:
        if prediction == 0:
            sentiments.append("Negative")
        elif prediction == 1:
            sentiments.append("Positive")

    return sentiments

