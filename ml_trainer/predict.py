from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import joblib
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
df = pd.read_csv("renderdata.csv")
print(df.columns)

# Select relevant columns and drop unnecessary ones
expected_columns = [
    'aa_samples', 'aov_count', 'file_size', 'frame_number', 'light_count',
    'polygon_count', 'resolution', 'production_label', 'task',
    'job_status', 'quality', 'render_time'
]

missing_columns = [col for col in expected_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing expected columns: {missing_columns}")

df = df[expected_columns]

# Drop rows with missing values
df.dropna(inplace=True)

# Convert resolution to total number of pixels
df['resolution'] = df['resolution'].apply(lambda x: int(x.split('x')[0]) * int(x.split('x')[1]))

# Encode categorical variables
label_encoders = {}
for column in ['production_label', 'task', 'quality']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Separate features and target variable
X = df.drop(['render_time', 'job_status', 'file_size', 'frame_number'], axis=1)
y = df['render_time']
print(X.columns)

# Normalize numerical features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

class RenderJobsDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create Dataset and DataLoader
dataset = RenderJobsDataset(X, y)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

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

input_size = X.shape[1]
model = RenderTimeModel(input_size)

# Define the optimizer with L2 regularization
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Training loop with visualization
num_epochs = 100
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_losses.append(running_loss / len(train_loader))

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

    test_losses.append(test_loss / len(test_loader))

    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_losses[-1]}, Test Loss: {test_losses[-1]}")

# Save the model
torch.save(model.state_dict(), 'render_time_model.pth')

# Save the label encoders and scaler
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Plot training and test loss
plt.figure()
plt.plot(range(num_epochs), train_losses, label='Training Loss')
plt.plot(range(num_epochs), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
