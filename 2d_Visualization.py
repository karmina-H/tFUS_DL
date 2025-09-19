import numpy as np
import matplotlib.pyplot as plt


data = np.load("./Data/output.npy")
data = data[0]

sum_axis0 = np.sum(data, axis=(1,2))
sum_axis1 = np.sum(data, axis=(0,2))
sum_axis2 = np.sum(data, axis=(0,1))

max_sums = [sum_axis0.max(),sum_axis1.max(), sum_axis2.max()]
max_axis = np.argmax(max_sums)
overall_max_sum = max_sums[max_axis]

if max_axis == 0:
    max_idx = np.argmax(sum_axis0)
    max_slice = data[max_idx, : :]
    title = f'Slice along X-axis at index {max_idx}'
    plt.xlabel('Y-axis')
    plt.ylabel('Z-axis')
elif max_axis == 1:
    max_idx = np.argmax(sum_axis1)
    max_slice = data[:, max_idx, :]
    title = f'Slice along Y-axis at index {max_idx}'
    plt.xlabel('X-axis')
    plt.ylabel('Z-axis')
else: # max_axis == 2
    max_idx = np.argmax(sum_axis2)
    max_slice = data[:, :, max_idx]
    title = f'Slice along Z-axis at index {max_idx}'
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

plt.figure(figsize=(10,8))
im = plt.imshow(max_slice, cmap='viridis', aspect='auto')

plt.colorbar(im, label='Value')
plt.title(f'{title}\nMax Sum: {overall_max_sum:.2f}')
plt.grid(False)
plt.show()

print(f"최대 합계를 가진 축: {['X', 'Y', 'Z'][max_axis]}")
print(f"해당 축의 인덱스: {max_idx}")
print(f"최대 합계 값: {overall_max_sum:.2f}")



# # 2. 각 축(0, 1, 2)을 기준으로 평면(slice)들의 합 계산
# sum_axis0 = np.sum(data, axis=(1, 2))  # X축 기준 (YZ 평면들)
# sum_axis1 = np.sum(data, axis=(0, 2))  # Y축 기준 (XZ 평면들)
# sum_axis2 = np.sum(data, axis=(0, 1))  # Z축 기준 (XY 평면들)

# # 3. 전체 데이터에서 가장 큰 합계 값을 가진 평면이 속한 '축' 찾기
# max_sum_overall = -1
# best_axis_index = -1
# target_sums = None

# if sum_axis0.max() > max_sum_overall:
#     max_sum_overall = sum_axis0.max()
#     best_axis_index = 0
#     target_sums = sum_axis0

# if sum_axis1.max() > max_sum_overall:
#     max_sum_overall = sum_axis1.max()
#     best_axis_index = 1
#     target_sums = sum_axis1

# if sum_axis2.max() > max_sum_overall:
#     max_sum_overall = sum_axis2.max()
#     best_axis_index = 2
#     target_sums = sum_axis2

# axis_names = ['X', 'Y', 'Z']
# best_axis_name = axis_names[best_axis_index]

# print(f"전체 데이터에서 가장 큰 합계({max_sum_overall:.2f})를 가진 평면은 '{best_axis_name}'축에 있습니다.")
# print("---")

# # 4. '가장 강한 축' 내에서 평균 계산 및 평균 이상인 평면들 필터링
# average_sum_in_best_axis = np.mean(target_sums)
# indices_to_plot = np.where(target_sums > average_sum_in_best_axis)[0]

# print(f"'{best_axis_name}'축 평면들의 평균 합계: {average_sum_in_best_axis:.2f}")
# print(f"평균보다 큰 합계를 가진 평면 인덱스: {indices_to_plot}\n")


# # 5. 해당되는 모든 평면을 하나씩 시각화
# for idx in indices_to_plot:
#     slice_sum = target_sums[idx]

#     if best_axis_index == 0: # X축
#         slice_data = data[idx, :, :]
#         title = f'Slice on Best Axis X at index {idx}'
#         plt.xlabel('Y-axis')
#         plt.ylabel('Z-axis')
#     elif best_axis_index == 1: # Y축
#         slice_data = data[:, idx, :]
#         title = f'Slice on Best Axis Y at index {idx}'
#         plt.xlabel('X-axis')
#         plt.ylabel('Z-axis')
#     else: # Z축
#         slice_data = data[:, :, idx]
#         title = f'Slice on Best Axis Z at index {idx}'
#         plt.xlabel('X-axis')
#         plt.ylabel('Y-axis')

#     plt.figure(figsize=(8, 6))
#     im = plt.imshow(slice_data, cmap='viridis', aspect='auto')
#     plt.colorbar(im, label='Value')
#     full_title = (f"{title}\n"
#                   f"Sum: {slice_sum:.2f} (Avg: {average_sum_in_best_axis:.2f})")
#     plt.title(full_title)
#     plt.grid(False)
#     plt.show()

