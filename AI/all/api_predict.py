import pickle
import torch
import torch.nn as nn
import numpy as np
import asyncio
from functools import lru_cache

# Model definition
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

_model_cache = {}

async def load_models():
    """Load all models and encoders (cached)"""
    if _model_cache:
        return _model_cache
    
    loop = asyncio.get_event_loop()
    
    tasks = [
        loop.run_in_executor(None, lambda: pickle.load(open("AI/all/Model/symptoms.pkl", "rb"))),
        loop.run_in_executor(None, lambda: pickle.load(open("AI/all/Model/disease_encoder.pkl", "rb"))),
        loop.run_in_executor(None, lambda: pickle.load(open("AI/all/Model/cause_encoder.pkl", "rb"))),
        loop.run_in_executor(None, lambda: pickle.load(open("AI/all/Model/prevention_encoder.pkl", "rb"))),
        loop.run_in_executor(None, lambda: torch.load("AI/all/Model/model.pth", map_location='cpu'))
    ]
    
    results = await asyncio.gather(*tasks)
    
    symptoms_list, disease_enc, cause_enc, prev_enc, checkpoint = results
    
    # Create model
    model = SymptomPredictor(
        checkpoint['input_size'],
        checkpoint['num_diseases'],
        checkpoint['num_causes'],
        checkpoint['num_preventions']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create symptom mapping
    symptom_to_idx = {s: i for i, s in enumerate(symptoms_list)}
    
    _model_cache.update({
        'model': model,
        'symptoms': symptoms_list,
        'symptom_to_idx': symptom_to_idx,
        'disease_enc': disease_enc,
        'cause_enc': cause_enc,
        'prevention_enc': prev_enc
    })
    
    return _model_cache

async def predict(symptom1: str, symptom2: str, symptom3: str) -> dict:
    """
    Async function to predict disease, cause, and prevention from symptoms
    
    Args:
        symptom1, symptom2, symptom3: Three symptoms as strings
    
    Returns:
        Dictionary with disease, cause, prevention and confidences
    """
    cache = await load_models()
    
    symptoms = [s.strip().lower() for s in [symptom1, symptom2, symptom3]]
    
    features = np.zeros(len(cache['symptoms']))
    valid_symptoms = []
    
    for s in symptoms:
        if s and s in cache['symptom_to_idx']:
            features[cache['symptom_to_idx'][s]] = 1
            valid_symptoms.append(s)
    
    # Predict
    with torch.no_grad():
        x = torch.FloatTensor(features).unsqueeze(0)
        d_out, c_out, p_out = cache['model'](x)
        
        # Get top predictions
        d_conf, d_pred = torch.max(torch.softmax(d_out, dim=1), 1)
        c_conf, c_pred = torch.max(torch.softmax(c_out, dim=1), 1)
        p_conf, p_pred = torch.max(torch.softmax(p_out, dim=1), 1)
        
        # Decode
        disease = cache['disease_enc'].inverse_transform([d_pred.item()])[0]
        cause = cache['cause_enc'].inverse_transform([c_pred.item()])[0]
        prevention = cache['prevention_enc'].inverse_transform([p_pred.item()])[0]
        
        return {
            'symptoms': valid_symptoms,
            'disease': disease,
            'disease_confidence': round(d_conf.item() * 100, 1),
            'cause': cause,
            'cause_confidence': round(c_conf.item() * 100, 1),
            'prevention': prevention,
            'prevention_confidence': round(p_conf.item() * 100, 1)
        }
