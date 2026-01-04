# 从300高度的无人机图像中提取第一张图像的标注，作为卫星图像的标注
import json

input_file = "../label-150-200.json"
output_file = "label-satellite.json"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

new_data = []

for item in data:
    # item 是一个字典：{path: caption}
    for path, caption in item.items():
        parts = path.split("/")

        # 结构：drone_view_512/0011/300/1.jpg
        if len(parts) != 4:
            continue

        root, scene, height, filename = parts

        # 只保留高度为 300 且文件名为 1.jpg
        if height == "300" and filename == "1.jpg":
            new_path = f"satellite-view/{scene}/0.png"
            new_data.append({new_path: caption})

# 写入新文件
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)

print(f"✅ 已完成转换，输出文件: {output_file}")
print(f"✅ 共生成 {len(new_data)} 条记录")
