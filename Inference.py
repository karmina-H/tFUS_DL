import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

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
    
dim_sizes = [6,24,48,96,48,24,4] # 처음 input dim부터 인덱스 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(dim_sizes).to(device) # -> 여기 MLP로 바꿔야함
model.load_state_dict(torch.load('your_model_weights.pth', map_location=torch.device(device)))

model.eval()
data = np.load("./Data/input.npy")
data = data[0]

result = model(data)

print(f'result : {result}')