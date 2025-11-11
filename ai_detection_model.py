# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from torch.optim import AdamW
import torch
from torch.utils.data import DataLoader, TensorDataset
import logging
import warnings
import os
from torch.cuda.amp import autocast, GradScaler


warnings.filterwarnings('ignore')
log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detection_model_log.txt")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_path, mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger()

# Custom callback for printing and logging
class TrainingLogger():
    def on_epoch_end(self, epoch, logs=None):
        log_message = f"Epoch {epoch + 1}: Loss = {logs['loss']:.4f}, Accuracy = {logs['accuracy']:.4f}, Validation Loss: {logs['val_loss']:.4f}, Validation Accuracy: {logs['val_accuracy']:.4f}"
        print(log_message)
        logger.info(log_message)

#Load the dataset
df = pd.read_csv("/mnt/c/ai_detection/ai_dataset/ai_human.csv")
logger.info("Dataset is loaded")
df = df.sample(n=80000, random_state=42)

#Split into training and testing datasets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
logger.info("Split into training and testing datasets")

# Check for class balance
sns.countplot(data=train_df, x=train_df['label'].astype(int))
plt.title("Training Dataset Class Balance")
plt.savefig('Training_Dataset_Class_Balance.png')
plt.clf()
sns.countplot(data=test_df, x=test_df['label'].astype(int))
plt.title("Testing Dataset Class Balance")
plt.savefig('Testing_Dataset_Class_Balance.png')
plt.clf()

# Split the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(train_df['text'], train_df['label'], test_size=0.2, random_state=42)
logger.info("Split into training and validation datasets")

# Tokenization and Encoding for DistilBERT
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
encoded_train = tokenizer(x_train.tolist(), padding=True, truncation=True, max_length=128, return_tensors='pt')
encoded_val = tokenizer(x_val.tolist(), padding=True, truncation=True, max_length=128, return_tensors='pt')

# Convert labels to tensors
train_labels = torch.tensor(y_train.values)
val_labels = torch.tensor(y_val.values)

# Create TensorDatasets
train_dataset = TensorDataset(encoded_train['input_ids'], encoded_train['attention_mask'], train_labels)
val_dataset = TensorDataset(encoded_val['input_ids'], encoded_val['attention_mask'], val_labels)

# DataLoader for efficient processing
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

logger.info("Training and validation datasets are ready for training process")

# Define the BERT model for sequence classification
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

logger.info("DistilBERT Model is loaded")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    logger.info(f"{device} GPU is being used")
else:
    logger.info("CPU is being used")
model.to(device)

# Define optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
epochs = 4

#Initialize GradScaler
scaler = GradScaler()

#Initialize logger class
training_logger = TrainingLogger()

#Metric storage
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

#Training and Validation loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    total_correct = 0

    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            preds = torch.argmax(outputs.logits, dim=1)
        total_loss += loss.item()
        total_correct += (preds == labels).sum().item()

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping to avoid exploding gradients
        scaler.step(optimizer)
        scaler.update()

    train_loss = total_loss / len(train_loader)
    train_acc = total_correct / len(train_dataset)

    model.eval()
    val_loss = 0
    val_correct = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            with autocast():
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                val_correct += (preds == labels).sum().item()
    val_loss /= len(val_loader)
    val_acc = val_correct / len(val_dataset)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    training_logger.on_epoch_end(epoch, {
        'loss': train_loss,
        'accuracy': train_acc,
        'val_loss': val_loss,
        'val_accuracy': val_acc
    })

#Visualization of training and validation process
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.title('Loss over Epochs')
plt.savefig('Loss_Curve.png')
plt.clf()
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')
plt.savefig('Accuracy_Curve.png')
plt.clf()

#Save the model
save_dir = "/mnt/c/ai_detection/saved_model"
os.makedirs(save_dir, exist_ok=True)
logger.info(f"Saving model and tokenizer to {save_dir}")
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

# Batched Testing
logger.info("Starting batch testing process")

test_encodings = tokenizer(
    test_df['text'].tolist(),
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors='pt'
)
test_labels = torch.tensor(test_df['label'].values)
test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], test_labels)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
model.eval()
preds_list = []
labels_list = []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        with autocast():
            outputs = model(input_ids, attention_mask=attention_mask)
            batch_preds = torch.argmax(outputs.logits, dim=1)
        preds_list.extend(batch_preds.cpu().numpy())
        labels_list.extend(labels.cpu().numpy())
preds = torch.tensor(preds_list)
test_labels = torch.tensor(labels_list)
test_acc = (preds == test_labels).sum().item() / len(test_labels)
logger.info(f"Test Accuracy: {test_acc:.4f}")

#Testing confusion matrix
cm = confusion_matrix(test_labels.cpu(), preds.cpu())
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Test Data)')
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.savefig('Test_Confusion_Matrix.png')
plt.clf()
