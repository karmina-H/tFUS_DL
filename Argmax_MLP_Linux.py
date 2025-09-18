import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch.nn as nn


# 데이터 로드
output_d = np.load("./Data/output.npy")
argmax_output = np.empty((len(output_d),4))

print(f'argmax_output len : {len(argmax_output)}')
for idx, data in enumerate(output_d):
    max_v = np.max(data)
    flatten_index = np.argmax(data)
    z,y,x = np.unravel_index(flatten_index, data.shape)
    argmax_output[idx] = [z,y,x, max_v]
    
print(f'argmax_output len : {len(argmax_output)}')
    
tensor_data_out = torch.from_numpy(argmax_output).float()

input_d = np.load("./Data/input.npy")
tensor_data_in = torch.from_numpy(input_d).float()


dataset = TensorDataset(tensor_data_in, tensor_data_out)

val_split = 0.1
dataset_size = len(dataset)
val_size = int(dataset_size * val_split)
train_size = dataset_size - val_size

torch.manual_seed(42)
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=False, num_workers=3)

vali_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=True, drop_last=False, num_workers=3)


# nn.Module을 상속받는 MLP 클래스 정의
class MLP(nn.Module):
    def __init__(self, dim_sizes):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim_sizes[0], dim_sizes[1]),
            nn.GELU(),
             nn.BatchNorm1d(dim_sizes[1]),
            nn.Dropout(0.4),

            nn.Linear(dim_sizes[1], dim_sizes[2]),
            nn.GELU(),
            nn.BatchNorm1d(dim_sizes[2]),
            nn.Dropout(0.3),

            nn.Linear(dim_sizes[2], dim_sizes[3]),
            nn.GELU(),
            nn.BatchNorm1d(dim_sizes[3]),
            nn.Dropout(0.2),

            nn.Linear(dim_sizes[3], dim_sizes[4]),
            nn.GELU(),
            nn.BatchNorm1d(dim_sizes[4]),

            nn.Linear(dim_sizes[4], dim_sizes[5]),
            nn.GELU(),
            nn.BatchNorm1d(dim_sizes[5]),

            nn.Linear(dim_sizes[5], dim_sizes[6])
        )
    
    # 결과 = 패치위치(3차원) + 초음파강도(raw value)(1차원) = 4차원벡터
    def forward(self, x):
        result = self.mlp(x)
        return result


# ----- 모델/손실/옵티마이저 -----
dim_sizes = [6,24,48,96,48,24,4] # 처음 input dim부터 인덱스 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(dim_sizes).to(device) # -> 여기 MLP로 바꿔야함
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 시각화용 list
train_loss_list = []
vali_loss_list = []

epochs = 3000

fig, ax = plt.subplots(figsize=(8,5))


for epoch in range(epochs):
    model.train()
    epoch_loss_train = 0.0
    epoch_loss_vali = 0.0

    for batch_in, batch_out in train_dataloader:
        batch_in, batch_out = batch_in.to(device), batch_out.to(device)
        optimizer.zero_grad()
        y_pred = model(batch_in)
        loss = criterion(y_pred, batch_out)
        loss.backward()
        optimizer.step()

        epoch_loss_train += loss.item()

    avg_train_loss = epoch_loss_train / len(train_dataloader)
            
    train_loss_list.append(avg_train_loss)

    if epoch % 100 == 0:
        torch.save(model.state_dict(), f"TCNN_RC{epoch}.pth")

    with torch.no_grad():
        model.eval()
        for batch_in, batch_out in vali_dataloader:
            batch_in, batch_out = batch_in.to(device), batch_out.to(device)
            y_pred = model(batch_in)
            val_loss = criterion(y_pred, batch_out)

            epoch_loss_vali += val_loss.item()


    avg_vali_loss = epoch_loss_vali / len(vali_dataloader)
    vali_loss_list.append(avg_vali_loss)

    # --- 실시간 그래프 업데이트 ---
    fig, ax = plt.subplots(figsize=(8,5))
    epochs_ran = range(len(train_loss_list))
    ax.plot(epochs_ran, train_loss_list, label="Training Loss", color="blue")
    ax.plot(epochs_ran, vali_loss_list, label="Validation Loss", color="red")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Simple_MLP Training Loss")
    ax.legend()
    ax.grid(True)

    # 그래프를 파일로 저장
    plt.savefig('training_loss_graph.png')
    print("그래프가 'training_loss_graph.png' 파일로 저장되었습니다.")
    # 훈련 손실과 검증 손실을 함께 출력하는 것이 좋습니다.
    print(f"[Epoch {epoch+1}/{epochs}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_vali_loss:.6f}")



