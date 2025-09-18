import numpy as np
import pandas as pd

try:
    arr_in = np.load('input.npy')

    # 위치 벡터와 방향 벡터 분리
    position_vectors = arr_in[:, :3]  # 위치 벡터 
    direction_vectors = arr_in[:, 3:] # 방향 벡터

    print("--- 방향 벡터 ---")

    magnitudes = np.linalg.norm(direction_vectors, axis=1)

    # 벡터가 정규화(normalized)되었는지 확인 (크기가 1에 가까운지)
    is_normalized = np.allclose(magnitudes, 1)

    if is_normalized:
        print("모든 방향 벡터가 정규화되었습니다 (벡터의 크기가 1에 가깝습니다).")
    else:
        print("일부 방향 벡터가 정규화되지 않았습니다.")
        print("\n방향 벡터 크기에 대한 요약 통계:")
        magnitude_stats = pd.Series(magnitudes).describe()
        print(magnitude_stats)

    # --- 위치 벡터 분석 ---
    print("\n--- 위치 벡터 분석 ---")
    print("위치 벡터의 각 좌표(X, Y, Z)에 대한 기술 통계:")

    position_df = pd.DataFrame(position_vectors, columns=['X', 'Y', 'Z'])
    print(position_df.describe())

except FileNotFoundError:
    print("오류: 'input.npy' 파일을 찾을 수 없습니다. 파일이 올바른 디렉토리에 있는지 확인해주세요.")
except Exception as e:
    print(f"오류가 발생했습니다: {e}")

'''
방향 벡터 크기에 대한 요약 통계:
count    1.125000e+03
mean     1.000015e+00
std      2.358833e-07
min      1.000015e+00
25%      1.000015e+00
50%      1.000015e+00
75%      1.000015e+00
max      1.000015e+00
--- 위치 벡터 분석 ---
위치 벡터의 각 좌표(X, Y, Z)에 대한 기술 통계:
                 X            Y            Z
count  1125.000000  1125.000000  1125.000000
mean    123.000000   111.000000   157.000000
std      14.148425    14.148425    14.148425
min     103.000000    91.000000   137.000000
25%     113.000000   101.000000   147.000000
50%     123.000000   111.000000   157.000000
75%     133.000000   121.000000   167.000000
max     143.000000   131.000000   177.000000
'''