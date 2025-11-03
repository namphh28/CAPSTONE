import json
from collections import Counter

# Đường dẫn tới file train annotation của iNaturalist 2018
json_path = "./data/train2018.json"

with open(json_path, 'r') as f:
    data = json.load(f)

# Đếm số ảnh theo class_id
class_counts = Counter(item['category_id'] for item in data['annotations'])

# Thống kê cơ bản
num_classes = len(class_counts)
num_tail = sum(1 for c in class_counts.values() if c < 20)
ratio_tail = num_tail / num_classes * 100

# Tìm class có nhiều và ít ảnh nhất
max_class = max(class_counts, key=class_counts.get)
min_class = min(class_counts, key=class_counts.get)

print(f"Tổng số lớp: {num_classes}")
print(f"Số lớp có <20 ảnh: {num_tail}")
print(f"Tỷ lệ: {ratio_tail:.2f}%")

print("\n📈 Class có nhiều ảnh nhất:")
print(f"  ID: {max_class}, Số ảnh: {class_counts[max_class]}")

print("\n📉 Class có ít ảnh nhất:")
print(f"  ID: {min_class}, Số ảnh: {class_counts[min_class]}")
