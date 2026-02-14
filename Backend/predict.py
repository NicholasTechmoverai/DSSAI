"""
Docstring for Backend.predict

"""
import joblib as jb 
ModelFileName =  "AI/model.h5"
Model = jb.load(ModelFileName)

async def Predict(X_features):
    Disease = Model.predict(X_features)
    return Disease
