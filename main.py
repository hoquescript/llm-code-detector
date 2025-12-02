"""
CodeGPTSensor Training Script - Python + Java
Distinguishing LLM-generated from Human-written Code
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os

# ========== Configuration ==========
class Config:
    # Paths
    python_data_path = "data/codegptsensor/python/train.jsonl"
    java_data_path = "data/codegptsensor/java/train.jsonl"
    save_dir = "models"
    
    # Model settings
    max_length = 256
    hidden_dim = 256
    batch_size = 64
    num_epochs = 5
    learning_rate = 0.001
    
    # Use full dataset
    use_sample = False
    sample_size_per_lang = 2500

# ========== Model Definition ==========
class CodeClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=2):
        super(CodeClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ========== Dataset ==========
class CodeDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

# ========== Main Training Function ==========
def main(config):
    print("="*60)
    print("CodeGPTSensor Training - Python + Java")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 1. Load Both Datasets
    print("\n[1/7] Loading datasets...")
    df_python = pd.read_json(config.python_data_path, lines=True)
    df_python['language'] = 'python'
    print(f"  Python: {len(df_python)} pairs")
    
    df_java = pd.read_json(config.java_data_path, lines=True)
    df_java['language'] = 'java'
    print(f"  Java: {len(df_java)} pairs")
    
    df = pd.concat([df_python, df_java], ignore_index=True)
    print(f"\nCombined: {len(df)} pairs")
    
    if config.use_sample:
        df = df.groupby('language').sample(n=config.sample_size_per_lang, random_state=42)
        print(f"Using sample: {len(df)} pairs")
    
    # 2. Load UniXcoder
    print("\n[2/7] Loading UniXcoder...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
    unixcoder = AutoModel.from_pretrained("microsoft/unixcoder-base")
    unixcoder.to(device)
    unixcoder.eval()
    print("✓ UniXcoder loaded")
    
    # 3. Extract Embeddings
    print("\n[3/7] Extracting embeddings...")
    embeddings_list = []
    labels_list = []
    
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df)):
            for code_text, label in [(row['code'], row['label']), 
                                      (row['contrast'], 1 - row['label'])]:
                try:
                    inputs = tokenizer(code_text, padding='max_length', 
                                      truncation=True, max_length=config.max_length,
                                      return_tensors="pt").to(device)
                    outputs = unixcoder(**inputs)
                    emb = outputs.last_hidden_state[:, 0, :].cpu().squeeze()
                    embeddings_list.append(emb)
                    labels_list.append(label)
                except Exception as e:
                    continue
    
    X = torch.stack(embeddings_list).numpy()
    y = labels_list
    print(f"✓ Embeddings: {X.shape}")
    
    # 4. Split Data
    print("\n[4/7] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(  
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    train_dataset = CodeDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_dataset = CodeDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # 5. Initialize Classifier
    print("\n[5/7] Initializing classifier...")
    classifier = CodeClassifier(input_dim=768, hidden_dim=config.hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=config.learning_rate)
    print(f"✓ Classifier ready")
    
    # 6. Training
    print(f"\n[6/7] Training {config.num_epochs} epochs...")
    best_acc = 0
    
    for epoch in range(config.num_epochs):
        classifier.train()
        train_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch_X, batch_y in pbar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            outputs = classifier(batch_X)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_acc = 100 * correct / total
        print(f"Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, Acc={train_acc:.2f}%")
        
        if train_acc > best_acc:
            best_acc = train_acc
            os.makedirs(config.save_dir, exist_ok=True)
            torch.save(classifier.state_dict(), f"{config.save_dir}/best_classifier.pt")
    
    # 7. Evaluation
    print("\n[7/7] Final evaluation...")
    classifier.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = classifier(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.numpy())
    
    test_acc = accuracy_score(all_labels, all_preds)
    print(f"\n{'='*60}")
    print(f"FINAL TEST ACCURACY: {test_acc*100:.2f}%")
    print(f"{'='*60}\n")
    print(classification_report(all_labels, all_preds, target_names=['Human', 'AI']))
    print(f"\n✓ Model saved to: {config.save_dir}/best_classifier.pt")

if __name__ == "__main__":
    config = Config()
    main(config)
