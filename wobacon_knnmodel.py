import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib  # For saving the model

# Function to train the model and save it
def train_and_save_model():
    # Download the dataset
    dataset = load_dataset('nesticot/mlb_data', data_files=['mlb_pitch_data_2023.csv',
                                                            'mlb_pitch_data_2022.csv',
                                                            'mlb_pitch_data_2021.csv',
                                                            'mlb_pitch_data_2020.csv'])
    dataset_train = dataset['train']

    # Convert dataset into Pandas DataFrame
    df = dataset_train.to_pandas().set_index(list(dataset_train.features.keys())[0]).reset_index(drop=True)

    # Preprocessing steps
    df['season'] = df['game_date'].str[0:4].astype(int)
    df['in_play'] = ['True' if x > 0 else np.nan for x in df['launch_speed']]

    conditions_tb = [(df['event_type'] == 'single'),
                     (df['event_type'] == 'double'),
                     (df['event_type'] == 'triple'),
                     (df['event_type'] == 'home_run')]
    choices_tb = [1, 2, 3, 4]
    df['tb'] = np.select(conditions_tb, choices_tb, default=0)

    conditions_woba = [(df['event_type'] == 'walk'),
                       (df['event_type'] == 'hit_by_pitch'),
                       (df['event_type'] == 'single'),
                       (df['event_type'] == 'double'),
                       (df['event_type'] == 'triple'),
                       (df['event_type'] == 'home_run')]
    choices_woba = [0.684, 0.715, 0.880, 1.261, 1.604, 2.085]
    df['woba'] = np.select(conditions_woba, choices_woba, default=0)

    df_bip = df[~df['in_play'].isnull()].dropna(subset=['launch_speed', 'launch_angle', 'in_play'])

    # Prepare data for training
    features = ['launch_angle', 'launch_speed']
    target = 'tb'

    df_model_bip_train = df_bip.dropna(subset=features + [target])

    X = df_model_bip_train[features]
    y = df_model_bip_train[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the KNN model
    model = KNeighborsClassifier(n_neighbors=12)
    model.fit(X_train, y_train)

    # Save the trained model using joblib
    joblib.dump(model, 'knn_model.pkl')

    print("Model trained and saved successfully!")

# Call this function when running the script to train and save the model
if __name__ == "__main__":
    train_and_save_model()
