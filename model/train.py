import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score

# Import custom loader
try:
    from model.preprocessing import load_data
except ModuleNotFoundError:
    from preprocessing import load_data

def train_model():
    """
    Main training execution function.
    1. Loads data
    2. Splits into train/test
    3. Builds a transformation pipeline 
    4. Trains a Random Forest
    5. Saves the model artifact
    """
    
    # 1. Load Data
    df = load_data()
    
    # 2. Split Features and Target
    X = df.drop('status', axis=1) # Features
    y = df['status']              # Target (0 = Good, 1 = Bad)

    # Define Feature Groups
    categorical_features = [
        'checkin_acc', 'credit_history', 'purpose', 'savings_acc', 
        'present_emp_since', 'personal_status', 'other_debtors', 
        'property', 'inst_plans', 'housing', 'job', 'telephone', 'foreign_worker'
    ]
    
    numerical_features = [
        'duration', 'amount', 'installment_rate', 'residing_since', 
        'age', 'num_credits', 'dependents'
    ]

    # 3. Define Preprocessing Pipelines
    
    # Numeric Transformer: Impute missing values with median, then scale
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical Transformer: Impute missing with most frequent, then OneHotEncode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine them into a single column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 4. Create the Full Pipeline
    
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])

    # 5. Split and Train
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training model (this might take a moment)...")
    clf.fit(X_train, y_train)

    # 6. Evaluate
    print("Evaluating model...")
    y_pred = clf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # 7. Save Model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, 'credit_risk_model.joblib')
    
    print(f"Saving model to {output_path}...")
    joblib.dump(clf, output_path)
    print("Done!")

if __name__ == "__main__":
    train_model()