import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from AutoEncoder import Autoencoder3D
import numpy as np

arr = np.load("../output.npy")

tensor_data = torch.from_numpy(arr)
dataset = TensorDataset(tensor_data, tensor_data)  # Autoencoder는 입력=출력
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# ----- 모델/손실/옵티마이저 -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder3D().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

loss_list = []


# ----- 학습 루프 -----
epochs = 300

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"[Epoch {epoch+1}/{epochs}] Loss: {epoch_loss/len(dataloader):.6f}")

# ----- 모델 저장 -----
torch.save(model.state_dict(), "autoencoder3d.pth")
