from sklearn.datasets import load_breast_cancer
from pathlib import Path

# Load dataset
dataset = load_breast_cancer(as_frame=True)

df = dataset.frame.drop(columns=["target"])

# Make malignant = 1 and benign = 0
df["target"] = (dataset.target == 0).astype(int)

# Save CSV
output_path = Path(__file__).parent / "dataset.csv"
df.to_csv(output_path, index=False)

print("Dataset saved successfully!")
print("Shape:", df.shape)
print("\nClass distribution:")
print(df["target"].value_counts())
print("\nMissing values:", df.isnull().sum().sum())
