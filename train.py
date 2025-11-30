import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os

#load dataset
df = pd.read_csv('data/kc_house_data.csv')