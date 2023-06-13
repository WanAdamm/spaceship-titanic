from sklearn.svm import NuSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

plt.style.use('fivethirtyeight')

df = pd.read_csv('train.csv')
df.dropna()
df = df.apply(LabelEncoder().fit_transform)  
df_test = pd.read_csv('test.csv')

relevant = ['Cabin', 'Age', "RoomService", "FoodCourt", "ShoppingMall", 'Spa', 'VRDeck']
X = df[relevant]
Y = df['Transported']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

clf = NuSVC()
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

score = accuracy_score(Y_test, Y_pred)
correctPrediction = accuracy_score(Y_test, Y_pred, normalize=False)
print("accuracy score:" , score, "with", correctPrediction, "correct prediction")