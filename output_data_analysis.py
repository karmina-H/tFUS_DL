import numpy as np
import time

# 파일 경로
file_path = 'output.npy'

# output파일에 있는 데이터는 음수가 없음.
# 대부분의 데이터가 0이고 특정 값만높음(초음파가 도달하는곳만)
# 그래서 이거 output data를 정규화 하고 안하고도 나중에 비교해보기

# 패치에서 가장 큰 value = 0.1036, 데이터 총개수 11억5천만개
# 그래서 norm을 구해도 값이 크게 나옴!

try:
    arr_mmap = np.load(file_path, mmap_mode='r') # mmap_mode로 파일을 열어 메모리 사용 최소화
    
    num_samples = arr_mmap.shape[0]
    
    total_sum = 0.0           # 모든 원소의 합
    total_sum_of_squares = 0.0 # 모든 원소의 제곱의 합
    total_count = arr_mmap.size # 전체 원소의 개수 -> 1159088625 = (1125*101*101*101)

    start_time = time.time()

    path_max = []

    for i in range(num_samples):
        sample_3d = arr_mmap[i]
        
        chunk_float = sample_3d

        # 현재 조각의 '합'과 '제곱의 합'을 전체 변수에 누적
        total_sum += np.sum(chunk_float)
        total_sum_of_squares += np.sum(np.square(chunk_float))
        path_max.append(np.max(chunk_float))
        
        if (i + 1) % 100 == 0:
            print(f"  {i + 1} / {num_samples} 처리 완료...")

    
    # 1. 평균 계산
    mean = total_sum / total_count
    
    # 2. 분산 계산
    variance = (total_sum_of_squares / total_count) - (mean ** 2)
    
    # 3. 표준편차 계산
    std_dev = np.sqrt(variance)
    
    # 4. Norm(크기) 계산 (추가된 부분)
    # Norm은 모든 원소의 제곱의 합에 루트를 씌운 값입니다.
    final_norm = np.sqrt(total_sum_of_squares)
    
    end_time = time.time()
    
    print("-" * 30)
    print(f"✅ 전체 데이터 평균 (Mean): {mean}")
    print(f"✅ 전체 데이터 표준편차 (Standard Deviation): {std_dev}")
    print(f"✅ 전체 데이터 Norm (크기): {final_norm}") 
    print(f"\n총 계산 소요 시간: {end_time - start_time:.2f} 초")

    print(max(path_max))

except FileNotFoundError:
    print(f"오류: '{file_path}' 파일을 찾을 수 없습니다.")
except Exception as e:
    print(f"오류가 발생했습니다: {e}")