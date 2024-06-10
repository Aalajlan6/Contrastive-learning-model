import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import MLP, GCNConv, global_max_pool
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV data
data = pd.read_csv('A_X_0.1.2.csv', skiprows=3)

# Assume the last column is the label and the rest are features
features = data.iloc[:, :-1]
labels = data.iloc[:, -1]

# Identify categorical and numerical columns
categorical_columns = features.select_dtypes(include=['object']).columns
numerical_columns = features.select_dtypes(exclude=['object']).columns

# One-Hot Encode categorical features
one_hot_encoder = OneHotEncoder(sparse_output=False)
categorical_encoded = one_hot_encoder.fit_transform(features[categorical_columns])

# Normalize numerical features
scaler = StandardScaler()
numerical_encoded = scaler.fit_transform(features[numerical_columns])

# Combine the encoded categorical and numerical features
X = np.hstack((numerical_encoded, categorical_encoded))
y = labels.values

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch Geometric format
def create_data(X, y):
    data_list = []
    for i in range(X.shape[0]):
        node_features = torch.tensor(X[i], dtype=torch.float)
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)  # Dummy edges for single node
        data = Data(x=node_features.unsqueeze(0), edge_index=edge_index, y=torch.tensor(y[i], dtype=torch.long))
        data_list.append(data)
    return data_list

train_data = create_data(X_train, y_train)
test_data = create_data(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Augmentation function
def augmentation(data):
    noise = torch.randn_like(data.x) * 0.1
    data.x = data.x + noise
    return data

# Contrastive Loss function
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, h_1, h_2, y):
        distances = (h_1 - h_2).pow(2).sum(1)  # Squared L2 distance
        losses = 0.5 * (y.float() * distances + (1 - y).float() * F.relu(self.margin - (distances + 1e-9).sqrt()).pow(2))
        return losses.mean()

# Model definition
class Model(torch.nn.Module):
    def __init__(self, k=20, aggr='max'):
        super().__init__()
        # Feature extraction
        self.conv1 = GCNConv(X_train.shape[1], 64)
        self.conv2 = GCNConv(64, 128)
        # Encoder head
        self.lin1 = Linear(128 + 64, 128)
        # Projection head (See explanation in SimCLRv2)
        self.mlp = MLP([128, 256, 128], norm=None)  # Changed final layer size to match 128

    def forward(self, data, train=True):
        if train:
            # Get 2 augmentations of the batch
            augm_1 = augmentation(data)
            augm_2 = augmentation(data)

            # Extract properties
            x1 = self.conv1(augm_1.x.squeeze(0), augm_1.edge_index)
            x2 = self.conv2(x1, augm_1.edge_index)
            h_points_1 = self.lin1(torch.cat([x1, x2], dim=1))

            x1 = self.conv1(augm_2.x.squeeze(0), augm_2.edge_index)
            x2 = self.conv2(x1, augm_2.edge_index)
            h_points_2 = self.lin1(torch.cat([x1, x2], dim=1))

            # Global representation
            h_1 = global_max_pool(h_points_1.unsqueeze(0), torch.tensor([0]))
            h_2 = global_max_pool(h_points_2.unsqueeze(0), torch.tensor([0]))

            # Transformation for loss function
            compact_h_1 = self.mlp(h_1)
            compact_h_2 = self.mlp(h_2)
            return h_1, h_2, compact_h_1, compact_h_2
        else:
            x1 = self.conv1(data.x.squeeze(0), data.edge_index)
            x2 = self.conv2(x1, data.edge_index)
            h_points = self.lin1(torch.cat([x1, x2], dim=1))
            h = global_max_pool(h_points.unsqueeze(0), torch.tensor([0]))
            compact_h = self.mlp(h)
            return h, compact_h

# Training and testing functions
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Increased learning rate
criterion = ContrastiveLoss(margin=1.0)

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        h_1, h_2, compact_h_1, compact_h_2 = model(data)
        # Assuming binary labels (1 for similar, 0 for dissimilar)
        y = torch.ones(h_1.size(0)).to(device)
        loss = criterion(compact_h_1, compact_h_2, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def test(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            h, compact_h = model(data, train=False)
            y = torch.ones(h.size(0)).to(device)
            loss = criterion(h, compact_h, y)
            total_loss += loss.item()
    return total_loss / len(loader)

# Training loop
epochs = 20
for epoch in range(epochs):
    train_loss = train(model, train_loader, optimizer, criterion, device)
    test_loss = test(model, test_loader, criterion, device)
    print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

    # Debugging: Check gradients and parameter updates
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f'Gradient for {name}: {param.grad.abs().mean()}')
        print(f'Parameter {name}: {param.data.abs().mean()}')
