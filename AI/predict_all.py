import pickle
import torch
import torch.nn as nn
import numpy as np
import os

# Configuration
MODEL_PATH = "AI/Model/model.pth"
DISEASE_ENCODER_PATH = "AI/Model/disease_encoder.pkl"
CAUSE_ENCODER_PATH = "AI/Model/cause_encoder.pkl"
PREVENTION_ENCODER_PATH = "AI/Model/prevention_encoder.pkl"
SYMPTOMS_PATH = "AI/Model/symptoms.pkl"

# Define the SAME model architecture used in training (with all 3 heads)
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

# Load artifacts
print("Loading model and encoders...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load symptoms list
with open(SYMPTOMS_PATH, 'rb') as f:
    unique_symptoms = pickle.load(f)

# Load encoders
with open(DISEASE_ENCODER_PATH, 'rb') as f:
    disease_encoder = pickle.load(f)
with open(CAUSE_ENCODER_PATH, 'rb') as f:
    cause_encoder = pickle.load(f)
with open(PREVENTION_ENCODER_PATH, 'rb') as f:
    prevention_encoder = pickle.load(f)

# Load model checkpoint
checkpoint = torch.load(MODEL_PATH, map_location=device)

# Initialize model with ALL heads
model = SymptomPredictor(
    input_size=checkpoint['input_size'],
    num_diseases=checkpoint['num_diseases'],
    num_causes=len(cause_encoder.classes_),
    num_preventions=len(prevention_encoder.classes_)
).to(device)

# Load state dict
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Create symptom to index mapping
symptom_to_idx = {symptom: i for i, symptom in enumerate(unique_symptoms)}

print(f"\n{'='*50}")
print("Disease Prediction from Symptoms")
print(f"{'='*50}\n")

# Prediction loop
while True:
    print("\nEnter 3 symptoms (or 'quit' to exit):")
    
    # Get symptoms from user
    symptom1 = input("Symptom 1: ").strip().lower()
    if symptom1 == 'quit':
        break
        
    symptom2 = input("Symptom 2: ").strip().lower()
    symptom3 = input("Symptom 3: ").strip().lower()
    
    # Create feature vector
    features = np.zeros(len(unique_symptoms))
    symptoms_entered = []
    unknown_symptoms = []
    
    for symptom in [symptom1, symptom2, symptom3]:
        if symptom:
            symptoms_entered.append(symptom)
            if symptom in symptom_to_idx:
                features[symptom_to_idx[symptom]] = 1
            else:
                unknown_symptoms.append(symptom)
    
    if unknown_symptoms:
        print(f"⚠️ Unknown symptoms: {', '.join(unknown_symptoms)}")
    
    if not any(features):
        print("⚠️ No valid symptoms entered!")
        continue
    
    # Predict
    with torch.no_grad():
        x = torch.FloatTensor(features).unsqueeze(0).to(device)
        disease_out, cause_out, prevention_out = model(x)
        
        # Get disease prediction
        disease_probs = torch.softmax(disease_out, dim=1)
        disease_conf, disease_idx = torch.max(disease_probs, 1)
        
        # Get cause prediction
        cause_probs = torch.softmax(cause_out, dim=1)
        cause_conf, cause_idx = torch.max(cause_probs, 1)
        
        # Get prevention prediction
        prevention_probs = torch.softmax(prevention_out, dim=1)
        prevention_conf, prevention_idx = torch.max(prevention_probs, 1)
        
        # Decode predictions
        disease = disease_encoder.inverse_transform([disease_idx.cpu().numpy()[0]])[0]
        cause = cause_encoder.inverse_transform([cause_idx.cpu().numpy()[0]])[0]
        prevention = prevention_encoder.inverse_transform([prevention_idx.cpu().numpy()[0]])[0]
        
        # Get top 3 diseases
        top_disease_probs, top_disease_indices = torch.topk(disease_probs, 3)
        
        print(f"\n{'='*60}")
        print(f"Symptoms: {', '.join(symptoms_entered)}")
        print(f"{'='*60}")
        
        # Main prediction
        print(f"\n🔍 DIAGNOSIS RESULTS")
        print(f"{'-'*60}")
        print(f"🏥 DISEASE: {disease.upper()}")
        print(f"   Confidence: {disease_conf.item()*100:.1f}%")
        print(f"\n🦠 CAUSE: {cause}")
        print(f"   Confidence: {cause_conf.item()*100:.1f}%")
        print(f"\n💊 PREVENTION: {prevention}")
        print(f"   Confidence: {prevention_conf.item()*100:.1f}%")
        
        # Alternative possibilities
        print(f"\n📋 OTHER POSSIBLE CONDITIONS:")
        print(f"{'-'*60}")
        for i in range(1, 3):
            alt_disease = disease_encoder.inverse_transform([top_disease_indices[0][i].cpu().numpy()])[0]
            alt_conf = top_disease_probs[0][i].item() * 100
            print(f"   {i}. {alt_disease}: {alt_conf:.1f}%")
        
        # Confidence indicator
        print(f"\n📊 CONFIDENCE LEVEL:")
        if disease_conf.item() * 100 > 70:
            print("   ✓ HIGH - Strong match with symptoms")
        elif disease_conf.item() * 100 > 40:
            print("   ⚠️ MEDIUM - Symptoms match multiple conditions")
        else:
            print("   ❓ LOW - Symptoms are very general")
    
    print(f"\n{'-'*60}")
    print("Press Enter for new prediction, or type 'quit'")

print("\n✅ Goodbye!")