import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch.nn as nn

def rc_decomposition_svd(voxel_grid, rank):
    """
    SVD를 사용하여 3D 복셀 데이터를 C와 R 행렬로 분해합니다.

    Args:
        voxel_grid (np.ndarray): (101, 101, 101) 형태의 3D 데이터.
        rank (int): 분해할 랭크(k). 압축률과 복원 품질을 결정합니다.

    Returns:
        tuple[np.ndarray, np.ndarray]: C 행렬과 R 행렬을 튜플 형태로 반환합니다.
    """
    # 1. 3D 복셀 데이터를 2D 행렬로 변환 (Matricization)
    # SVD는 2D 행렬에 대해서만 연산이 가능합니다.
    # (101, 101, 101) -> (101, 10201)
    original_shape = voxel_grid.shape
    matrix_a = voxel_grid.reshape(original_shape[0], -1)

    # 2. 특잇값 분해(SVD) 수행
    # A ≈ U @ S @ Vh
    # full_matrices=False 옵션은 연산 속도를 향상시킵니다.
    U, S, Vh = np.linalg.svd(matrix_a, full_matrices=False)

    # 3. 정의한 rank(k) 만큼 행렬들을 잘라냅니다.
    U_k = U[:, :rank]
    S_k = np.diag(S[:rank])
    Vh_k = Vh[:rank, :]

    # 4. C와 R 행렬 정의
    # C 행렬: 데이터의 핵심 구조(basis)를 담고 있습니다.
    # R 행렬: 이 구조들을 어떻게 조합할지에 대한 계수(coefficients)를 담고 있습니다.
    # C @ R 을 통해 원래 행렬 matrix_a를 근사적으로 복원할 수 있습니다.
    C = U_k                 # Shape: (101, rank)
    R = S_k @ Vh_k          # Shape: (rank, 10201)

    return C, R


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# RC-Decomposition 이용해서 데이터 압축 및 저장하는 코드
output_d = np.load("./Data/output.npy")

list_of_C = []
list_of_R = []

RANK = 10

for idx, data in enumerate(output_d): # output_d = (1152,101,101,101)
    C, R = rc_decomposition_svd(data, RANK)
    list_of_C.append(C)
    list_of_R.append(R)

output_filename = f'decomposed_data_rank_{RANK}.npz'
np.savez_compressed(
    output_filename,
    C_matrices=np.array(list_of_C),
    R_matrices=np.array(list_of_R)
)
    

batch_size = 32


# 데이터 로드

output_d = np.load(f"./Data/decomposed_data_rank_{RANK}.npz")

# R을 예측하는 것
output_R = output_d['R_matrices']
output_C = output_d['C_matrices']

print(output_R.shape)
print(output_C.shape)

tensor_data_out = torch.from_numpy(output_R).float()

input_d = np.load("./Data/input.npy")
tensor_data_in = torch.from_numpy(input_d).float()

dataset = TensorDataset(tensor_data_in, tensor_data_out)

val_split = 0.1
dataset_size = len(dataset)
val_size = int(dataset_size * val_split)
train_size = dataset_size - val_size

torch.manual_seed(42)
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=3)

vali_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=3)


import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=6, output_channels=10, output_features=10201):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.output_channels = output_channels
        self.output_features_per_channel = output_features

        # 목표 출력 크기
        self.final_output_size = output_channels * output_features

        # 1. 초기 MLP: 6개의 입력 값을 1024개로 확장
        self.initial_mlp = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
        )

        # (batch, 1024) -> reshape to (batch, 256, 2, 2)
        # 2x2 크기에서 시작하여 32x32까지 점진적으로 확장
        self.upsample_blocks = nn.Sequential(
            # 입력: (batch, 256, 2, 2)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # -> (batch, 128, 4, 4)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # -> (batch, 64, 8, 8)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # -> (batch, 32, 16, 16)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), # -> (batch, 16, 32, 32)
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
        )

        # 3. 최종 MLP: 업샘플링된 피처를 최종 출력 크기로 매핑
        # (16 * 32 * 32 = 16384) -> (10 * 10201 = 102010)
        self.final_mlp = nn.Sequential(
            nn.Linear(16 * 32 * 32, 4096),
            nn.LeakyReLU(0.2),
            nn.Linear(4096, self.final_output_size),
            # 마지막 활성화 함수는 문제에 따라 tanh, sigmoid 등을 추가할 수 있습니다.
            # nn.Tanh()
        )


    def forward(self, x):
        # 1. 초기 MLP 통과
        out = self.initial_mlp(x) # (batch, 1024)

        # 2. ConvTranspose2d를 위한 형태로 변경
        # (batch, 256, 2, 2)
        out = out.view(out.size(0), 256, 2, 2)

        # 3. 업샘플링 블록 통과
        out = self.upsample_blocks(out) # (batch, 16, 32, 32)

        # 4. 최종 MLP를 위한 형태로 변경
        out = out.view(out.size(0), -1) # (batch, 16*32*32)

        # 5. 최종 MLP 통과
        out = self.final_mlp(out) # (batch, 10 * 10201)

        # 6. 최종 출력 형태로 변경
        out = out.view(out.size(0), self.output_channels, self.output_features_per_channel) # (batch, 10, 10201)

        return out
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Generator().to(device)
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

    if epoch % 50 == 0:
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

# ----- 모델 저장 -----
torch.save(model.state_dict(), "TCNN_RC.pth")
