

import torch
import torch.nn as nn

import pandas as pd

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

data = pd.read_csv("Heart_Disease_Prediction.csv")
print(data.head())

# Unique output values
unique_output = data["Heart Disease"].unique()
print(unique_output)


y = data["Heart Disease"].map({"Presence": 1, "Absence": 0}).values
X = data.drop("Heart Disease", axis=1).values

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size = 16, shuffle = True)


model = nn.Sequential(
        nn.Linear(13, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
        nn.Sigmoid()
        )

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)


# Training loop
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")


model.eval()  # evaluation mode
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_class = (y_pred >= 0.5).float()
    accuracy = (y_pred_class == y_test).sum() / y_test.size(0)
    print(f"Test Accuracy: {accuracy:.4f}")

