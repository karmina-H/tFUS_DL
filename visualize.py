import numpy as np
import plotly.graph_objects as go
import os

# --- 설정 ---
file_path = 'output.npy'
# 시각화할 샘플 인덱스
sample_index = 0
# 다운샘플링 비율 (2이면 2칸마다 하나씩, 즉 해상도를 1/8로 줄임)
downsample_rate = 2

# --- 데이터 로드 ---
try:
    arr_mmap = np.load(file_path, mmap_mode='r')
    print(f"샘플 {sample_index}의 3D 데이터를 로드합니다...")
    sample_3d = arr_mmap[500]

    # --- 다운샘플링 ---
    data_downsampled = sample_3d[::downsample_rate, ::downsample_rate, ::downsample_rate]
    print(f"원본 데이터 크기: {sample_3d.shape}")
    print(f"다운샘플링된 데이터 크기: {data_downsampled.shape}")

    # --- 3D 볼륨 시각화 ---
    X, Y, Z = np.mgrid[:data_downsampled.shape[0], :data_downsampled.shape[1], :data_downsampled.shape[2]]

    fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=data_downsampled.flatten(),
        isomin=0.0005,  # 최소 임계값 (이 값보다 작은 데이터는 투명하게 처리)
        isomax=1.1,  # 최대 임계값
        opacity=0.3, # 전체적인 불투명도
        surface_count=300, # 표면 개수 (많을수록 디테일하지만 느려짐)
        colorscale='Viridis'
    ))

    fig.update_layout(scene_xaxis_showticklabels=False,
                      scene_yaxis_showticklabels=False,
                      scene_zaxis_showticklabels=False)

    print("인터랙티브 3D 시각화를 생성합니다... (브라우저나 창이 열릴 수 있습니다)")
    fig.show()

    # HTML 파일로 저장하여 공유 가능
    if not os.path.exists("3d_visualization"):
        os.makedirs("3d_visualization")
    fig.write_html("3d_visualization/sample_0_3d_volume.html")
    print("결과가 HTML 파일로 저장되었습니다.")

except Exception as e:
    print(f"오류가 발생했습니다: {e}")