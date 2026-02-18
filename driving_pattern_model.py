import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

def create_dummy_data(num_samples=1000):
    """
    Creates dummy dataset for training the driving pattern ML model.
    Features:
    - average_speed (km/h)
    - sudden_brakes (count per trip)
    - driving_duration (minutes)
    - time_of_day (hour 0-23)
    - alertness_score (output from drowsy detection model 0-100)
    
    Target labels:
    0: Safe
    1: Rash / Aggressive
    2: Drowsy
    """
    np.random.seed(42)
    
    # Safe drivers (label 0)
    safe_speed = np.random.normal(60, 10, int(num_samples * 0.5))
    safe_brakes = np.random.poisson(1, int(num_samples * 0.5))
    safe_dur = np.random.normal(45, 15, int(num_samples * 0.5))
    safe_time = np.random.randint(6, 20, int(num_samples * 0.5))
    safe_alert = np.random.normal(90, 5, int(num_samples * 0.5))
    
    # Rash drivers (label 1)
    rash_speed = np.random.normal(90, 15, int(num_samples * 0.3))
    rash_brakes = np.random.poisson(5, int(num_samples * 0.3))
    rash_dur = np.random.normal(30, 20, int(num_samples * 0.3))
    rash_time = np.random.randint(0, 24, int(num_samples * 0.3))
    rash_alert = np.random.normal(85, 10, int(num_samples * 0.3))
    
    # Drowsy drivers (label 2)
    drowsy_speed = np.random.normal(50, 15, int(num_samples * 0.2))
    drowsy_brakes = np.random.poisson(2, int(num_samples * 0.2))
    drowsy_dur = np.random.normal(120, 60, int(num_samples * 0.2))
    drowsy_time = np.random.choice([0, 1, 2, 3, 4, 22, 23], int(num_samples * 0.2))
    drowsy_alert = np.random.normal(50, 20, int(num_samples * 0.2))
    
    speed = np.concatenate([safe_speed, rash_speed, drowsy_speed])
    brakes = np.concatenate([safe_brakes, rash_brakes, drowsy_brakes])
    dur = np.concatenate([safe_dur, rash_dur, drowsy_dur])
    time = np.concatenate([safe_time, rash_time, drowsy_time])
    alert = np.concatenate([safe_alert, rash_alert, drowsy_alert])
    
    labels = np.concatenate([np.zeros(int(num_samples*0.5)), 
                             np.ones(int(num_samples*0.3)), 
                             np.ones(int(num_samples*0.2))*2])
                             
    df = pd.DataFrame({
        'average_speed': np.clip(speed, 0, 150),
        'sudden_brakes': np.clip(brakes, 0, 20),
        'driving_duration': np.clip(dur, 0, 600),
        'time_of_day': time,
        'alertness_score': np.clip(alert, 0, 100),
        'pattern': labels
    })
    
    return df

def train_model():
    print("Generating simulated driver data...")
    df = create_dummy_data(2000)
    
    X = df.drop('pattern', axis=1)
    y = df['pattern']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    print("Training Random Forest model on driving patterns...")
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    print("\nModel Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    target_names = ['Safe (0)', 'Rash/Aggressive (1)', 'Drowsy (2)']
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    print("\nFeature Importances:")
    importances = clf.feature_importances_
    for name, imp in zip(X.columns, importances):
        print(f"  {name}: {imp:.4f}")
        
    print("\nSaving model to driving_pattern_rf.pkl...")
    joblib.dump(clf, 'driving_pattern_rf.pkl')
    print("Done!")

if __name__ == "__main__":
    train_model()
