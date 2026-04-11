import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import argparse

def train(n_estimators):
    # Load data
    data_path = "namadataset_preprocessing/credit_risk_processed.csv"
    df = pd.read_csv(data_path)
    
    X = df.drop(columns=['loan_status'])
    y = df['loan_status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        # Explicitly log to ensure we hit criteria
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("accuracy", acc)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        print(f"Model trained with n_estimators={n_estimators}, Accuracy={acc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    args = parser.parse_args()
    
    train(args.n_estimators)
