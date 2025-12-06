import pandas as pd
import sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import login, HfApi

# Login to Hugging Face
login(os.getenv("HF_TOKEN"))
api = HfApi(token=os.getenv("HF_TOKEN"))

DATASET_PATH = "hf://datasets/neerajig/Tourism-Package-Prediction/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop the unique identifier and the first unnamed column if it exists
df.drop(columns=['CustomerID'], inplace=True, errors='ignore')
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

# Encoding the categorical 'TypeofContact' column
label_encoder = LabelEncoder()
df['TypeofContact'] = label_encoder.fit_transform(df['TypeofContact'])

target_col = 'ProdTaken'
X = df.drop(columns=[target_col])
y = df[target_col]

# Clean 'Gender' and 'MaritalStatus' columns
df['Gender'] = df['Gender'].str.replace("Fe Male", "FeMale")
df['MaritalStatus'] = df['MaritalStatus'].str.replace("Unmarried", "Single")

# Train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save files
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# Upload files to the correct repo
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id="neerajig/Tourism-Package-Prediction",
        repo_type="dataset",
    )
