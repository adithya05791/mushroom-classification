import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(filepath):
    # Load dataset
    data = pd.read_csv(filepath)
    
    # Encode categorical features to numerical
    label_encoder = LabelEncoder()
    for column in data.columns:
        data[column] = label_encoder.fit_transform(data[column])
    
    # Split into features and target
    X = data.drop('class', axis=1)
    y = data['class']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, X.columns