# predict_simple.py - Run with: python predict_simple.py fever cough fatigue
import pickle
import torch
import torch.nn as nn
import numpy as np
import sys

# Define model (same as training)
class SymptomPredictor(nn.Module):
    def __init__(self, input_size, num_diseases, num_causes, num_preventions):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
        )
        self.disease_head = nn.Linear(32, num_diseases)
        self.cause_head = nn.Linear(32, num_causes)
        self.prevention_head = nn.Linear(32, num_preventions)
    
    def forward(self, x):
        features = self.network(x)
        return self.disease_head(features), self.cause_head(features), self.prevention_head(features)

# Load everything
print("Loading model...")
symptoms = pickle.load(open("AI/all/Model/symptoms.pkl", "rb"))
disease_enc = pickle.load(open("AI/all/Model/disease_encoder.pkl", "rb"))
cause_enc = pickle.load(open("AI/all/Model/cause_encoder.pkl", "rb"))
prev_enc = pickle.load(open("AI/all/Model/prevention_encoder.pkl", "rb"))
checkpoint = torch.load("AI/all/Model/model.pth", map_location='cpu')

model = SymptomPredictor(
    checkpoint['input_size'], 
    checkpoint['num_diseases'],
    checkpoint['num_causes'],
    checkpoint['num_preventions']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Get symptoms from command line or input
if len(sys.argv) > 1:
    input_symptoms = [s.lower() for s in sys.argv[1:4]]
else:
    input_symptoms = input("Enter 3 symptoms (comma separated): ").lower().split(',')

# Create feature vector
symptom_to_idx = {s: i for i, s in enumerate(symptoms)}
features = np.zeros(len(symptoms))
valid_symptoms = []
for s in input_symptoms[:3]:
    s = s.strip()
    if s in symptom_to_idx:
        features[symptom_to_idx[s]] = 1
        valid_symptoms.append(s)

# Predict
with torch.no_grad():
    d_out, c_out, p_out = model(torch.FloatTensor(features).unsqueeze(0))
    
    d_conf, d_pred = torch.max(torch.softmax(d_out, dim=1), 1)
    c_conf, c_pred = torch.max(torch.softmax(c_out, dim=1), 1)
    p_conf, p_pred = torch.max(torch.softmax(p_out, dim=1), 1)
    
    disease = disease_enc.inverse_transform([d_pred.item()])[0]
    cause = cause_enc.inverse_transform([c_pred.item()])[0]
    prevention = prev_enc.inverse_transform([p_pred.item()])[0]

# Print results
print(f"\n{'='*50}")
print(f"Symptoms: {', '.join(valid_symptoms)}")
print(f"{'='*50}")
print(f"DISEASE: {disease}")
print(f"Confidence: {d_conf.item()*100:.1f}%")
print(f"\nCAUSE: {cause}")
print(f"\nPREVENTION: {prevention}")
print(f"{'='*50}")