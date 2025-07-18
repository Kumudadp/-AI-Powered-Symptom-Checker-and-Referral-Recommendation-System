import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import joblib

# Load the dataset
df = pd.read_csv("AI_Symptom_Checker_Dataset.csv")

# Strip column names in case of spaces
df.columns = df.columns.str.strip()

# Parse and preprocess the symptoms column
df['Symptoms'] = df['Symptoms'].str.strip().str.lower()
df['Symptoms'] = df['Symptoms'].str.split(',')

# Convert symptom list to binary features (multi-hot encoding)
mlb = MultiLabelBinarizer()
symptom_features = pd.DataFrame(mlb.fit_transform(df['Symptoms']), columns=mlb.classes_)

# Combine with other features if needed
X = symptom_features

# Encode target column
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Predicted Disease'])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model, label encoder, and symptom feature transformer
joblib.dump(model, "model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(mlb, "symptom_encoder.pkl")

print("✅ Model trained and saved as model.pkl, label_encoder.pkl, and symptom_encoder.pkl")
