import numpy as np

# 1. 파일 경로 설정
input_file = './Data/input.npy'
output_file = 'input_first_3.npy'  # 원본을 덮어쓰지 않도록 새 이름으로 저장

try:
    # 2. npy 파일 불러오기
    print(f"'{input_file}' 파일을 불러오는 중...")
    data = np.load(input_file)
    print(f"불러온 원본 데이터의 형태: {data.shape}")

    # 3. 데이터 3개만 선택하기 (슬라이싱)
    first_three_items = data[:3]
    print(f"앞에서 3개만 선택한 데이터의 형태: {first_three_items.shape}")

    # 4. 선택한 데이터를 새로운 npy 파일로 저장하기
    np.save(output_file, first_three_items)
    print(f"✅ 성공! 3개의 데이터가 '{output_file}' 파일로 저장되었습니다.")

except FileNotFoundError:
    print(f"❌ 에러: '{input_file}' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
except Exception as e:
    print(f"❌ 에러가 발생했습니다: {e}")