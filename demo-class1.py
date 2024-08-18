import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Load and preprocess data
file_path = 'C:/Users/navin/OneDrive/Pictures/Breast_cancer_Project/data.csv'
df = pd.read_csv(file_path)

# Drop unnecessary columns
df = df.drop(columns=['id', 'Unnamed: 32'])

# Map target variable to binary values
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Define features and target
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=102)

# Preprocessing steps
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Record preprocessing steps
preprocessing_steps = {
    "scaling": {
        "method": "StandardScaler",
        "mean": scaler.mean_.tolist(),  # mean used for scaling
        "var": scaler.var_.tolist(),    # variance used for scaling
    },
    "train_test_split": {
        "test_size": 0.2,
        "random_state": 102
    }
}

# Function to log preprocessing steps
def log_preprocessing(preprocessing_steps):
    mlflow.log_dict(preprocessing_steps, "preprocessing_steps.json")

# Define a function to handle model training, evaluation, and logging
def train_evaluate_log_model(model, model_name, X_train, X_test, y_train, y_test, params, preprocessing_steps):
    # Log preprocessing steps
    log_preprocessing(preprocessing_steps)

    # Train the model
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    print(f"{model_name}:")
    print(classification_report(y_test, y_pred))

    # Generate classification report as a dictionary
    class_report = classification_report(y_test, y_pred, output_dict=True)

    # Log with MLflow
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics({
            'accuracy': class_report['accuracy'],
            'recall_class_0': class_report['0']['recall'],
            'recall_class_1': class_report['1']['recall'],
            'f1_score': class_report['macro avg']['f1-score']
        })
        mlflow.sklearn.log_model(model, model_name)
        print(f"Model {model_name} logged successfully.\n")

# Set up MLflow experiment
experiment_name = "breast cancer_data"
mlflow.set_experiment("breast cancer_data")
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000/")

# Logistic Regression
logistic_params = {
    "solver": "lbfgs",
    "max_iter": 10000,
    "multi_class": "auto",
    "random_state": 8888,
}
logistic_model = LogisticRegression(**logistic_params)
train_evaluate_log_model(logistic_model, "Logistic Regression", X_train, X_test, y_train, y_test, logistic_params, preprocessing_steps)

# Decision Tree Classifier
dt_params = {
    "random_state": 8888,
}
dt_model = DecisionTreeClassifier(**dt_params)
train_evaluate_log_model(dt_model, "Decision Tree Classifier", X_train, X_test, y_train, y_test, dt_params, preprocessing_steps)

# Additional models can be added similarly:
# Example:
# from sklearn.ensemble import RandomForestClassifier
# rf_params = {"n_estimators": 100, "random_state": 8888}
# rf_model = RandomForestClassifier(**rf_params)
# train_evaluate_log_model(rf_model, "Random Forest Classifier", X_train, X_test, y_train, y_test, rf_params, preprocessing_steps)
