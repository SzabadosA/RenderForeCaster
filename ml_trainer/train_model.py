from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import joblib
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import optuna
from sklearn.model_selection import train_test_split

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Is CUDA available? ", torch.cuda.is_available())
print("CUDA version: ", torch.version.cuda)
print("Number of CUDA devices: ", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Current CUDA device: ", torch.cuda.current_device())
    print("CUDA device name: ", torch.cuda.get_device_name(torch.cuda.current_device()))

# Read the CSV file into a DataFrame
df = pd.read_csv("renderdata.csv")
print(df.columns)
criterion = nn.MSELoss()
# Select relevant columns and drop unnecessary ones
expected_columns = [
    "aa_samples",
    "aov_count",
    "file_size",
    "frame_number",
    "light_count",
    "polygon_count",
    "resolution",
    "production_label",
    "task",
    "job_status",
    "quality",
    "render_time",
]

missing_columns = [col for col in expected_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing expected columns: {missing_columns}")

df = df[expected_columns]

# Drop rows with missing values
df.dropna(inplace=True)

# Convert resolution to total number of pixels
df["resolution"] = df["resolution"].apply(
    lambda x: int(x.split("x")[0]) * int(x.split("x")[1])
)

# Encode categorical variables
label_encoders = {}
for column in ["production_label", "task", "quality"]:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Separate features and target variable
X = df.drop(["render_time", "job_status", "file_size", "frame_number"], axis=1)
y = df["render_time"]
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

def objective(trial):
    # Define the hyperparameters to tune
    input_size = X.shape[1]
    hidden_size1 = trial.suggest_int('hidden_size1', 32, 256)
    hidden_size2 = trial.suggest_int('hidden_size2', 16, 128)
    hidden_size3 = trial.suggest_int('hidden_size3', 8, 64)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)

    model = RenderTimeModel(input_size, hidden_size1, hidden_size2, hidden_size3, dropout_rate).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    # Train/test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = RenderJobsDataset(X_train, y_train)
    val_dataset = RenderJobsDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    # Training loop
    num_epochs = 50
    best_val_loss = float('inf')
    early_stop_counter = 0
    early_stop_patience = 5

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= early_stop_patience:
            break

    return best_val_loss

# Create a study and optimize the objective function
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print('Best trial:')
trial = study.best_trial
print('  Value: ', trial.value)
print('  Params: ')
for key, value in trial.params.items():
    print(f'    {key}: {value}')

# Get the best hyperparameters
best_params = study.best_trial.params

# Initialize the model with the best hyperparameters
model = RenderTimeModel(
    input_size=X.shape[1],
    hidden_size1=best_params['hidden_size1'],
    hidden_size2=best_params['hidden_size2'],
    hidden_size3=best_params['hidden_size3'],
    dropout_rate=best_params['dropout_rate']
).to(device)

# Define the optimizer with the best learning rate
optimizer = optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=1e-5)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# Training loop with early stopping and learning rate scheduling
num_epochs = 100
train_losses = []
test_losses = []
best_test_loss = float('inf')
early_stop_patience = 10
early_stop_counter = 0

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

train_dataset = RenderJobsDataset(X_train, y_train)
val_dataset = RenderJobsDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
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
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

    test_losses.append(test_loss / len(test_loader))
    scheduler.step(test_loss / len(test_loader))

    print(
        f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_losses[-1]}, Test Loss: {test_losses[-1]}"
    )

    # Early stopping
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        early_stop_counter = 0
        # Save the best model
        torch.save(model.state_dict(), "best_render_time_model.pth")
    else:
        early_stop_counter += 1

    if early_stop_counter >= early_stop_patience:
        print("Early stopping triggered")
        break

# Save the final model
torch.save(model.state_dict(), "final_render_time_model.pth")

# Save the label encoders and scaler
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(scaler, "scaler.pkl")

# Plot training and test loss
plt.figure()
plt.plot(range(len(train_losses)), train_losses, label="Training Loss")
plt.plot(range(len(test_losses)), test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
