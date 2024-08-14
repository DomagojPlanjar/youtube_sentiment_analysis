import os

import torch
from sklearn.metrics import accuracy_score
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, get_scheduler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import time

# HyperParameters
MODEL_NAME = 'distilbert-base-uncased'
MAX_LEN = 200
BATCH_SIZE = 64
EPOCHS = 4
LR = 5e-5
EARLY_STOPPING_PATIENCE = 1
EARLY_STOPPING_THRESHOLD = 1e-4

# Class label mapping
CLASS_NAMES = {0: 'Negative', 1: 'Positive'}
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    # Load the data
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(current_dir)

    # Change between 'data.csv' and 'balanced_downsampled_data.csv' as wished
    file_path = os.path.join(parent_dir, 'data', 'balanced_downsampled_data.csv')
    data = pd.read_csv(file_path, encoding='ISO-8859-1')

    # Shuffle the dataset
    df_shuffled = data.sample(frac=1).reset_index(drop=True)

    # Initialize tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # Move model to GPU
    model.to(device)

    # Extract texts and labels
    texts = df_shuffled.iloc[:, -1].tolist()
    labels = df_shuffled.iloc[:, 0].apply(lambda x: 0 if x == 0 else 1).tolist()

    # Split data into training and test sets before tokenization
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    print("Splitting data finished")

    def tokenize(texts, tokenizer, max_len):
        return tokenizer(texts, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')

    # Tokenize both train and test sets
    train_inputs = tokenize(train_texts, tokenizer, MAX_LEN)
    test_inputs = tokenize(test_texts, tokenizer, MAX_LEN)
    print("Tokenizing finished")

    # Move training and testing tensors to GPU
    train_input_ids = train_inputs['input_ids'].to(device)
    train_attention_masks = train_inputs['attention_mask'].to(device)
    train_labels = torch.tensor(train_labels).to(device)

    test_input_ids = test_inputs['input_ids'].to(device)
    test_attention_masks = test_inputs['attention_mask'].to(device)
    test_labels = torch.tensor(test_labels).to(device)
    print("Data to GPU finished")

    # Create DataLoader
    train_data = TensorDataset(train_input_ids, train_attention_masks, train_labels)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    test_data = TensorDataset(test_input_ids, test_attention_masks, test_labels)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    print("Dataloader finished")

    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # Define the learning rate scheduler
    num_training_steps = len(train_loader) * EPOCHS
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup

    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        # Track time
        start_time = time.time()

        for index, batch in enumerate(train_loader):
            batch_input_ids, batch_attention_mask, batch_labels = batch
            print(f"{index + 1}. Batch of {epoch + 1}. epoch in training")
            optimizer.zero_grad()
            outputs = model(
                batch_input_ids,
                attention_mask=batch_attention_mask,
                labels=batch_labels
            )

            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Collect predictions and labels for accuracy calculation
            # (to track possible overfitting while training)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(train_loader)
        avg_accuracy = accuracy_score(all_labels, all_preds)
        print(f"Finished training epoch {epoch + 1}/{EPOCHS} in {epoch_time:.2f} seconds - Training Loss: {avg_loss:.4f} - Training Accuracy: {avg_accuracy:.4f}")

        val_preds = []
        val_labels = []

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                batch_input_ids, batch_attention_mask, batch_labels = batch
                outputs = model(
                    batch_input_ids,
                    attention_mask=batch_attention_mask,
                    labels=batch_labels
                )
                val_loss = outputs.loss
                total_val_loss += val_loss.item()

                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1)
                val_preds.extend(predictions.cpu().numpy())
                val_labels.extend(batch_labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(test_loader)
        avg_accuracy = accuracy_score(val_labels, val_preds)

        print(f"Validation Loss after epoch {epoch + 1}: {avg_val_loss:.4f} -  Validation Accuracy: {avg_accuracy:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss - EARLY_STOPPING_THRESHOLD:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save the best model
            model_save_path = os.path.join(parent_dir, 'models', 'distilbert_model_best.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f"Best model saved to {model_save_path}")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    # Save the final model
    model_save_path = os.path.join(parent_dir, 'models', 'distilbert_model_final.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Final model saved to {model_save_path}")

    # Evaluation
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in test_loader:
            batch_input_ids, batch_attention_mask, batch_labels = batch
            outputs = model(
                batch_input_ids,
                attention_mask=batch_attention_mask
            )
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == batch_labels).sum().item()
            total_predictions += batch_labels.size(0)

    accuracy = correct_predictions / total_predictions
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Define a function for text classification
    def classify_text(text, model, tokenizer, max_len):
        model.eval()
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()

        return CLASS_NAMES[predicted_class]

    # Interactive classification to test out the model
    print("Enter sentences to classify (type 'exit' to quit):")
    while True:
        user_input = input("Sentence: ")
        if user_input.lower() == 'exit':
            break
        predicted_class = classify_text(user_input, model, tokenizer, MAX_LEN)
        print(f"Predicted class: {predicted_class}")


if __name__ == '__main__':
    main()
