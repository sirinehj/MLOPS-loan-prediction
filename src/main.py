import pandas as pd
from model import prepare_data, train, evaluate, save_model, save_metrics

def main():
    # Load data
    df = pd.read_csv("data/loan_data.csv")
    
    # Prepare data
    df_clean = prepare_data(df)
    
    # Train the model
    model, X_test, y_test, X_train, y_train = train(df_clean)
    
    # Evaluate the model
    metrics = evaluate(model, X_test, y_test)
    
    # Save the model and metrics
    save_model(model, "models/svc_model.pkl")
    save_metrics(metrics, "models/svc_metrics.json")

if __name__ == "__main__":
    main()