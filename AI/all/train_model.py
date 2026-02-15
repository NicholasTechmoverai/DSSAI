import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os

DATA_PATH = "AI/disease_prediction_dataset.csv"
SAVE_PATH = "AI/all/Model"
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 0.001
TEST_SIZE = 0.15

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n{'='*50}")
print(f"Using device: {device}")
print(f"{'='*50}\n")

df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded {len(df)} samples")

# Clean data
for col in ['symptom1', 'symptom2', 'symptom3', 'disease', 'cause', 'prevention']:
    df[col] = df[col].astype(str).str.lower().str.strip()

# 2. CREATE FEATURE MATRIX (One-hot encoding)
all_symptoms = []
for col in ['symptom1', 'symptom2', 'symptom3']:
    all_symptoms.extend(df[col].tolist())
unique_symptoms = sorted(list(set(all_symptoms)))
symptom_to_idx = {s: i for i, s in enumerate(unique_symptoms)}

X = []
for _, row in df.iterrows():
    features = np.zeros(len(unique_symptoms))
    for col in ['symptom1', 'symptom2', 'symptom3']:
        symptom = row[col]
        if symptom in symptom_to_idx:
            features[symptom_to_idx[symptom]] = 1
    X.append(features)
X = np.array(X)

# 3. ENCODE TARGETS
disease_encoder = LabelEncoder()
cause_encoder = LabelEncoder()
prevention_encoder = LabelEncoder()

y_disease = disease_encoder.fit_transform(df['disease'])
y_cause = cause_encoder.fit_transform(df['cause'])
y_prevention = prevention_encoder.fit_transform(df['prevention'])

print(f"\nDataset Info:")
print(f"  Symptoms: {len(unique_symptoms)} unique")
print(f"  Diseases: {len(disease_encoder.classes_)}")
print(f"  Causes: {len(cause_encoder.classes_)}")
print(f"  Preventions: {len(prevention_encoder.classes_)}")

# Check class distribution
disease_counts = pd.Series(y_disease).value_counts()
print(f"\nClass Distribution:")
print(f"  Classes with 1 sample: {sum(disease_counts == 1)}")
print(f"  Classes with 2+ samples: {sum(disease_counts > 1)}")

# 4. TRAIN/TEST SPLIT
split = train_test_split(# Clean da# Clean data
ta

    X, y_disease, y_cause, y_prevention,
    test_size=TEST_SIZE, 
    random_state=42
)
X_train, X_test, y_disease_train, y_disease_test, y_cause_train, y_cause_test, y_prevention_train, y_prevention_test = split

print(f"\nSplit: {len(X_train)} train, {len(X_test)} test")
print(f"  Unique diseases in train: {len(np.unique(y_disease_train))}")
print(f"  Unique diseases in test: {len(np.unique(y_disease_test))}")

# 5. DATASET CLASS
class SymptomDataset(Dataset):
    def __init__(self, X, y_disease, y_cause, y_prevention):
        self.X = torch.FloatTensor(X)
        self.y_disease = torch.LongTensor(y_disease)
        self.y_cause = torch.LongTensor(y_cause)
        self.y_prevention = torch.LongTensor(y_prevention)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_disease[idx], self.y_cause[idx], self.y_prevention[idx]

# Create data loaders
train_loader = DataLoader(SymptomDataset(X_train, y_disease_train, y_cause_train, y_prevention_train), 
                         batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(SymptomDataset(X_test, y_disease_test, y_cause_test, y_prevention_test), 
                        batch_size=BATCH_SIZE)

class SymptomPredictor(nn.Module):
    def __init__(self, input_size, num_diseases, num_causes, num_preventions):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.disease_head = nn.Linear(32, num_diseases)
        self.cause_head = nn.Linear(32, num_causes)
        self.prevention_head = nn.Linear(32, num_preventions)
    
    def forward(self, x):
        features = self.network(x)
        return (self.disease_head(features), 
                self.cause_head(features), 
                self.prevention_head(features))

# Initialize model
model = SymptomPredictor(
    input_size=len(unique_symptoms),
    num_diseases=len(disease_encoder.classes_),
    num_causes=len(cause_encoder.classes_),
    num_preventions=len(prevention_encoder.classes_)
).to(device)

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

# TRAINING SETUP
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"\n{'='*50}")
print("STARTING TRAINING")
print(f"{'='*50}\n")

best_acc = 0
train_losses = []
test_accs = []

for epoch in range(EPOCHS):
    # Training
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for X_batch, y_d, y_c, y_p in train_loader:
        X_batch, y_d, y_c, y_p = X_batch.to(device), y_d.to(device), y_c.to(device), y_p.to(device)
        
        optimizer.zero_grad()
        d_out, c_out, p_out = model(X_batch)
        
        loss = (criterion(d_out, y_d) + 0.2 * criterion(c_out, y_c) + 0.2 * criterion(p_out, y_p))
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(d_out, 1)
        total += y_d.size(0)
        correct += (predicted == y_d).sum().item()
    
    train_acc = 100 * correct / total
    avg_loss = train_loss / len(train_loader)
    train_losses.append(avg_loss)
    
    # Testing
    model.eval()
    correct = 0
    total = 0
    all_correct = 0
    
    with torch.no_grad():
        for X_batch, y_d, y_c, y_p in test_loader:
            X_batch, y_d, y_c, y_p = X_batch.to(device), y_d.to(device), y_c.to(device), y_p.to(device)
            d_out, c_out, p_out = model(X_batch)
            
            _, pred_d = torch.max(d_out, 1)
            _, pred_c = torch.max(c_out, 1)
            _, pred_p = torch.max(p_out, 1)
            
            total += y_d.size(0)
            correct += (pred_d == y_d).sum().item()
            all_correct += ((pred_d == y_d) & (pred_c == y_c) & (pred_p == y_p)).sum().item()
    
    test_acc = 100 * correct / total
    test_accs.append(test_acc)
    
    # Save best model
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'best_model.pth')
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.1f}% | Test Acc: {test_acc:.1f}%")

print(f"\n{'='*50}")
print(f"✅ Training Complete!")
print(f"Best Test Accuracy: {best_acc:.1f}%")
print(f"{'='*50}\n")

model.load_state_dict(torch.load('best_model.pth'))

print("\n" + "="*50)
print("SAMPLE PREDICTIONS")
print("="*50)

model.eval()
correct = 0
confidences = []

with torch.no_grad():
    for i in range(min(10, len(X_test))):
        x = torch.FloatTensor(X_test[i]).unsqueeze(0).to(device)
        
        # Get symptoms
        symptom_indices = np.where(X_test[i] > 0)[0]
        symptoms = [unique_symptoms[idx] for idx in symptom_indices]
        
        # Predict ALL outputs
        d_out, c_out, p_out = model(x)
        
        # Disease prediction
        d_probs = torch.softmax(d_out, dim=1)
        d_conf, d_pred = torch.max(d_probs, 1)
        
        # Cause prediction
        c_probs = torch.softmax(c_out, dim=1)
        c_conf, c_pred = torch.max(c_probs, 1)
        
        # Prevention prediction
        p_probs = torch.softmax(p_out, dim=1)
        p_conf, p_pred = torch.max(p_probs, 1)
        
        # Decode
        actual_disease = disease_encoder.inverse_transform([y_disease_test[i]])[0]
        predicted_disease = disease_encoder.inverse_transform([d_pred.cpu().numpy()[0]])[0]
        predicted_cause = cause_encoder.inverse_transform([c_pred.cpu().numpy()[0]])[0]
        predicted_prevention = prevention_encoder.inverse_transform([p_pred.cpu().numpy()[0]])[0]
        
        is_correct = actual_disease == predicted_disease
        if is_correct:
            correct += 1
            confidences.append(d_conf.item() * 100)
        
        print(f"\n{i+1}. Symptoms: {', '.join(symptoms[:3])}")
        print(f"   Actual Disease: {actual_disease}")
        print(f"   Predicted: {predicted_disease} ({d_conf.item()*100:.1f}%) {'✅' if is_correct else '❌'}")
        print(f"   Cause: {predicted_cause} ({c_conf.item()*100:.1f}%)")
        print(f"   Prevention: {predicted_prevention} ({p_conf.item()*100:.1f}%)")

# SUMMARY
print(f"\n{'='*50}")
print("SUMMARY")
print(f"{'='*50}")
print(f"Best Test Accuracy: {best_acc:.1f}%")
if confidences:
    print(f"Avg Confidence (correct): {np.mean(confidences):.1f}%")
print(f"Sample correct: {correct}/10")

# SAVE MODEL AND ALL ENCODERS
os.makedirs(SAVE_PATH, exist_ok=True)

# Save model with ALL dimensions
torch.save({
    'model_state_dict': model.state_dict(),
    'input_size': len(unique_symptoms),
    'num_diseases': len(disease_encoder.classes_),
    'num_causes': len(cause_encoder.classes_),
    'num_preventions': len(prevention_encoder.classes_)
}, os.path.join(SAVE_PATH, 'model.pth'))

# Save ALL encoders
with open(os.path.join(SAVE_PATH, 'disease_encoder.pkl'), 'wb') as f:
    pickle.dump(disease_encoder, f)
with open(os.path.join(SAVE_PATH, 'cause_encoder.pkl'), 'wb') as f:
    pickle.dump(cause_encoder, f)
with open(os.path.join(SAVE_PATH, 'prevention_encoder.pkl'), 'wb') as f:
    pickle.dump(prevention_encoder, f)
with open(os.path.join(SAVE_PATH, 'symptoms.pkl'), 'wb') as f:
    pickle.dump(unique_symptoms, f)

print(f"\n✅ Model and all encoders saved to {SAVE_PATH}")
print(f"{'='*50}\n")