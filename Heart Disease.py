import numpy as np
from xgboost import XGBClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

file = pd.read_csv("heart.csv")
dummies = ["ChestPainType","RestingECG", "ExerciseAngina","Sex","ST_Slope"] # Pick the collums with Multiple Classes
pd.set_option('display.max_columns', None) #
onehotencoded = pd.get_dummies(file,dummies) # One Hot Encode Collums

prediction = onehotencoded["HeartDisease"]
onehotencoded = onehotencoded.drop(["HeartDisease"], axis = 1 )
x_train,x_test,y_train,y_test = train_test_split(onehotencoded,prediction,train_size=0.8,shuffle= True)
x_train,x_val, y_train, y_val = train_test_split(x_train,y_train,train_size = 0.8 )
# x_train, y_train dados para treinar, x_test e y_test dados para testar, x_val e y_val dados para cross validar modelos
model = XGBClassifier(n_estimators = 10000, learning_rate = 0.001,early_stopping_rounds=10) # Initialize the  classifier
model.fit(x_train,y_train,eval_set=[(x_val,y_val)])
prediction = model.predict(x_test)
train_accuracy = accuracy_score(y_test, prediction)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

# Accuracy between 85% and 90% More Work needed