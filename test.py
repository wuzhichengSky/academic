import json

# 1. 读取原始标注文件
with open("./merge/merged.json", "r", encoding="utf-8") as f:
    annotations = json.load(f)

train_data = []
test_data = []

# 2. 按 scene id 划分
for item in annotations:
    # drone_view_512/0001/150/0.jpg → 0001
    scene_id = int(item["drone"].split("/")[1])

    if scene_id <= 160:
        train_data.append(item)
    else:
        test_data.append(item)

# 3. 写出 train.json
with open("train.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

# 4. 写出 test.json
with open("test.json", "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

print(f"Train samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")
