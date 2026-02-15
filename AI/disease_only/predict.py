import pickle
import torch
import torch.nn as nn
import numpy as np
import os

# Configuration
MODEL_PATH = "AI/disease_only/Model/model.pth"
ENCODER_PATH = "AI/disease_only/Model/disease_encoder.pkl"
SYMPTOMS_PATH = "AI/disease_only/Model/symptoms.pkl"

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
        return self.disease_head(features)  # We only need disease for prediction

# Load artifacts
print("Loading model and encoders...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load symptoms list
with open(SYMPTOMS_PATH, 'rb') as f:
    unique_symptoms = pickle.load(f)

# Load disease encoder
with open(ENCODER_PATH, 'rb') as f:
    disease_encoder = pickle.load(f)

# Load model checkpoint
checkpoint = torch.load(MODEL_PATH, map_location=device)

# Initialize model with ALL heads (matching training)
model = SymptomPredictor(
    input_size=checkpoint['input_size'],
    num_diseases=checkpoint['num_diseases'],
    num_causes=checkpoint.get('num_causes', 63),  # Default from your data
    num_preventions=checkpoint.get('num_preventions', 117)  # Default from your data
).to(device)

# Load state dict (strict=False ignores missing keys)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()

symptom_to_idx = {symptom: i for i, symptom in enumerate(unique_symptoms)}

print(f"\n{'='*50}")
print("Disease Prediction from Symptoms")
print(f"{'='*50}\n")

while True:
    print("\nEnter 3 symptoms (or 'quit' to exit):")
    
    symptom1 = input("Symptom 1: ").strip().lower()
    if symptom1 == 'quit':
        break
        
    symptom2 = input("Symptom 2: ").strip().lower()
    symptom3 = input("Symptom 3: ").strip().lower()
    
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
    
    # Predict
    with torch.no_grad():
        x = torch.FloatTensor(features).unsqueeze(0).to(device)
        output = model(x)
        probabilities = torch.softmax(output, dim=1)
        
        # Get top 3 predictions
        top_probs, top_indices = torch.topk(probabilities, 3)
        
        print(f"\n{'='*50}")
        print(f"Symptoms: {', '.join(symptoms_entered)}")
        print(f"{'='*50}")
        print("\nTop 3 Predictions:")
        
        predictions = []
        for i in range(3):
            disease = disease_encoder.inverse_transform([top_indices[0][i].cpu().numpy()])[0]
            confidence = top_probs[0][i].item() * 100
            predictions.append((disease, confidence))
            print(f"  {i+1}. {disease}: {confidence:.1f}%")
        
        # Top prediction details
        top_disease, top_confidence = predictions[0]
        
        print(f"\n👉 Most likely: {top_disease} ({top_confidence:.1f}% confidence)")
        
        if top_confidence > 70:
            print("   ✓ High confidence")
        elif top_confidence > 40:
            print("   ⚠️ Medium confidence")
        else:
            print("   ❓ Low confidence (symptoms may be too general)")
    
    print(f"\n{'-'*50}")
    print("Options: press Enter for new prediction, or type 'quit'")
    
