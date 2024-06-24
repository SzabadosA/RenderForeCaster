import pandas as pd
import torch
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


class RenderJobsDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).view(-1, 1) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]


class RenderTimeModel(nn.Module):
    def __init__(self, input_size):
        super(RenderTimeModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def load_model_and_data():
    # Load the trained model
    input_size = 8  # Adjust based on the number of features
    model = RenderTimeModel(input_size)
    model.load_state_dict(torch.load("render_time_model.pth"))
    model.eval()

    # Load the label encoders and scaler
    label_encoders = joblib.load("label_encoders.pkl")
    scaler = joblib.load("scaler.pkl")

    return model, label_encoders, scaler


def preprocess_input(data, label_encoders, scaler):
    # Convert resolution to total number of pixels
    data["resolution"] = data["resolution"].apply(lambda x: int(x.split("x")[0]) * int(x.split("x")[1]))

    # Encode categorical variables
    for column in ["production_label", "task", "quality"]:
        data[column] = label_encoders[column].transform(data[column])

    # Select relevant columns
    print(data)
    #data = data.drop(["render_time", "job_status", "file_size", "frame_number"], axis=1)

    # Normalize numerical features
    data = pd.DataFrame(scaler.transform(data), columns=data.columns)

    return data


def predict(model, data):
    dataset = RenderJobsDataset(data)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    predictions = []
    with torch.no_grad():
        for inputs in data_loader:
            outputs = model(inputs)
            predictions.append(outputs.item())

    return predictions


if __name__ == "__main__":
    # Load the model, encoders, and scaler
    model, label_encoders, scaler = load_model_and_data()

    # Example input data
    data = pd.DataFrame({
        "aa_samples": [1],
        "aov_count": [7],
        "light_count": [8],
        "polygon_count": [271940],
        "resolution": ["1000x500"],
        "production_label": ["fx_complex"],
        "task": ["compositing"],
        "quality": ["final"]
    })

    # Preprocess the input data
    preprocessed_data = preprocess_input(data, label_encoders, scaler)

    # Make predictions
    predictions = predict(model, preprocessed_data)

    # Print predictions
    print("Predicted render time:", predictions)
