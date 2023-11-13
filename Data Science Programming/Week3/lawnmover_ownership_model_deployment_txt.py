import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pickle


owner_model = pickle.load(open('./winning_model.pkl', "rb"))

print("\n*****************************************************")
print("* The Lawn Mover Ownership Prediction Model *")
print("*****************************************************\n")
income = float(input("Enter the income: "))
lot_size = float(input("Enter the Lotsize: "))
data={'Income':[income],'Lot_Size':[lot_size]}
df=pd.DataFrame(data)
result = owner_model.predict(df)
probability = owner_model.predict_proba(df)
print(f"\nThe Lawn Mover Ownership model indicates probability of owning a lawn mover at {probability[0][1]:.4f}, therefore it's indicated this property is {result[0]}.\n")


