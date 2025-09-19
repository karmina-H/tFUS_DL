import numpy as np
import pyvista as pv

# --- 데이터 로드 ---
# 'output.npy' 파일에서 NumPy 배열 데이터를 불러옵니다.
# 이 파일은 3차원 배열 형태의 데이터라고 가정합니다.
try:
    data = np.load('./Data/output.npy')
except FileNotFoundError:
    print("오류: 'output.npy' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
    # 예시용 랜덤 데이터 생성
    print("예시용 랜덤 데이터를 생성하여 시각화를 진행합니다.")
    data = np.random.rand(50, 50, 50)

data = data[0]
# --- PyVista 데이터 객체 생성 ---
# NumPy 배열을 PyVista가 이해할 수 있는 StructuredGrid 형태로 변환합니다.
# StructuredGrid는 데이터가 규칙적인 격자 구조를 가질 때 사용합니다.
grid = pv.StructuredGrid()

# 데이터의 차원(x, y, z)에 맞게 그리드의 크기를 설정합니다.
grid.dimensions = data.shape

# 그리드의 각 점(꼭짓점)의 좌표를 생성합니다.
# 여기서는 0부터 시작하는 정수 좌표를 사용합니다.
grid.points = np.vstack(
    [arr.ravel() for arr in np.meshgrid(
        np.arange(data.shape[0]),
        np.arange(data.shape[1]),
        np.arange(data.shape[2]),
        indexing='ij'
    )]
).T

# 그리드의 각 셀(cell) 또는 포인트(point)에 스칼라 값을 할당합니다.
# 이 값이 색상 등으로 시각화에 표현됩니다.
# 데이터의 순서를 'F'(포트란 순서)로 변경하여 PyVista의 데이터 구조와 일치시킵니다.
grid['values'] = data.flatten(order='F')


# --- 시각화 ---
# PyVista 플로터(Plotter) 객체를 생성합니다.
plotter = pv.Plotter()

# 플로터에 볼륨(volume)을 추가하여 데이터를 3D로 렌더링합니다.
# 볼륨 렌더링은 데이터의 내부 구조를 반투명하게 보여주는 기법입니다.
# 'cmap' 옵션으로 컬러맵을 지정할 수 있습니다. (예: 'viridis', 'plasma', 'coolwarm')
plotter.add_volume(grid, cmap='viridis')

# 시각화 창의 제목을 설정합니다.
plotter.camera_position = 'iso'
plotter.window_size = [800, 800]
plotter.add_axes()
plotter.show_bounds(all_edges=True)


# 시각화 창을 엽니다.
print("3D 시각화 창을 엽니다. 창을 닫으면 코드가 종료됩니다.")
plotter.show()

print("시각화가 종료되었습니다.")