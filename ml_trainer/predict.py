import torch
import joblib
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt

# Define the model class if not already defined
class RenderTimeModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, dropout_rate):
        super(RenderTimeModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the best hyperparameters
best_params = joblib.load("best_params.pkl")

# Load the trained model
model = RenderTimeModel(
    input_size=8,  # Update with the correct input size
    hidden_size1=best_params['hidden_size1'],
    hidden_size2=best_params['hidden_size2'],
    hidden_size3=best_params['hidden_size3'],
    dropout_rate=best_params['dropout_rate']
).to(device)

model.load_state_dict(torch.load("best_render_time_model.pth"))
model.eval()

# Load the label encoders and scaler
label_encoders = joblib.load("label_encoders.pkl")
scaler = joblib.load("scaler.pkl")

# Example new data
new_data = {
    "aa_samples": 8,
    "aov_count": 4,
    "light_count": 10,
    "polygon_count": 500000,
    "resolution": "1920x1080",
    "production_label": "fx_easy",
    "task": "lighting",
    "quality": "high"
}

# Convert to DataFrame
new_df = pd.DataFrame([new_data])

# Convert resolution to total number of pixels
new_df["resolution"] = new_df["resolution"].apply(
    lambda x: int(x.split("x")[0]) * int(x.split("x")[1])
)

# Encode categorical variables
for column in ["production_label", "task", "quality"]:
    le = label_encoders[column]
    new_df[column] = le.transform(new_df[column])

# Normalize numerical features
new_X = pd.DataFrame(scaler.transform(new_df), columns=new_df.columns)

# Convert to PyTorch tensor
new_X_tensor = torch.tensor(new_X.values, dtype=torch.float32).to(device)

# Make prediction
with torch.no_grad():
    prediction = model(new_X_tensor)

# Convert prediction to numpy array and print
predicted_render_time = prediction.cpu().numpy().flatten()
print("Predicted render time:", predicted_render_time)

# Testing gradual changes
resolutions = ["1920x1080", "2560x1440", "3840x2160"]
predictions = []

for res in resolutions:
    test_data = new_data.copy()
    test_data["resolution"] = res
    test_df = pd.DataFrame([test_data])
    test_df["resolution"] = test_df["resolution"].apply(lambda x: int(x.split("x")[0]) * int(x.split("x")[1]))
    for column in ["production_label", "task", "quality"]:
        le = label_encoders[column]
        test_df[column] = le.transform(test_df[column])
    test_X = pd.DataFrame(scaler.transform(test_df), columns=test_df.columns)
    test_X_tensor = torch.tensor(test_X.values, dtype=torch.float32).to(device)
    with torch.no_grad():
        pred = model(test_X_tensor)
    predictions.append(pred.cpu().numpy().flatten()[0])

plt.plot(resolutions, predictions, marker='o')
plt.xlabel("Resolution")
plt.ylabel("Predicted Render Time")
plt.title("Predicted Render Time vs. Resolution")
plt.show()
