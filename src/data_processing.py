import pandas as pd
import numpy as np
from pathlib import Path

def load_day_length_data(filepath=None):
    if filepath is None:
        filepath = Path('data/data.csv')
    
    data = pd.read_csv(filepath)
    return data

def convert_minutes_to_hours(data):
    data = data.copy()
    data['duration_hours'] = data['duration_minutes'] / 60
    data['duration_hours_decimal'] = data['duration_hours']
    
    hours = data['duration_minutes'] // 60
    minutes = data['duration_minutes'] % 60
    data['duration_formatted'] = hours.astype(str) + 'h ' + minutes.astype(str) + 'm'
    
    return data

def prepare_model_data(data):
    x = data['day_number'].values.astype(float)
    y = data['duration_minutes'].values.astype(float)
    
    return x, y

data = load_day_length_data()
print("Data loaded successfully!")
print(f"Shape: {data.shape}")
print(f"First 5 rows:\n{data.head()}")

data = convert_minutes_to_hours(data)
print(f"\nWith hours conversion:\n{data[['day_number', 'duration_minutes', 'duration_hours', 'duration_formatted']].head()}")
