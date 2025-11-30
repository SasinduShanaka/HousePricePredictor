import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os

#load dataset
df = pd.read_csv('data/kc_house_data.csv')

#keep only these columns
df = df[["price", "sqft_living", "bedrooms", "bathrooms"]]    

#x and y split
X = df[["sqft_living", "bedrooms", "bathrooms"]]
y = df["price"]

#train test split
X_train,  X_test, y_train.y_test = train_test_split(X,y,test_size=0.2, random_state=42)

#train model
model = LinearRegression()
model.fit(X_train,y_train)
print("Model trained successfully.")

#quick evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
rmse = mean_squared_error(y_test, preds, squared=False)
r2 = r2_score(y_test, preds)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2: {r2:.3f}")

#save the model
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'model/model.pkl')
print("Model saved to model/model.pkl")
