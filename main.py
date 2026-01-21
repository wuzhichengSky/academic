import json
import os

# =========================
# 配置路径和阈值
# =========================
INPUT_JSON_PATH = "./merge/merged_denseUAV.json"
TRAIN_JSON_PATH = "train.json"
TEST_JSON_PATH = "test.json"

# 分割阈值：ID <= 2427 为训练集，其余为测试集
SPLIT_THRESHOLD = 2427


def load_json(path: str):
    """读取 JSON 文件"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str):
    """保存 JSON 文件"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def get_scene_id_int(path: str) -> int:
    """
    从路径中提取场景ID并转换为整数。
    
    输入示例: "drone/000000/H80.JPG"
    逻辑:
      1. 统一路径分隔符为 '/'
      2. 分割字符串
      3. 提取中间的文件夹名 (即 000000)
    """
    # 将反斜杠替换为正斜杠，以防 Windows 路径问题
    parts = path.replace("\\", "/").split("/")
    
    # 假设结构固定为: 根目录/场景ID/文件名
    # 例如: parts[0]="drone", parts[1]="000000", parts[2]="H80.JPG"
    if len(parts) < 2:
        raise ValueError(f"Invalid path format: {path}")
        
    scene_id_str = parts[1]
    
    # 将字符串转为整数，自动处理前导零 (int("000000") -> 0)
    return int(scene_id_str)


def main():
    try:
        data = load_json(INPUT_JSON_PATH)
    except FileNotFoundError:
        print(f"Error: 找不到输入文件 {INPUT_JSON_PATH}")
        return

    train_data = []
    test_data = []

    print(f"正在处理 {len(data)} 条数据...")

    for item in data:
        drone_path = item.get("drone", "")
        
        if not drone_path:
            print(f"Warning: item missing 'drone' key: {item}")
            continue

        try:
            sid = get_scene_id_int(drone_path)
            
            if sid <= SPLIT_THRESHOLD:
                train_data.append(item)
            else:
                test_data.append(item)
                
        except ValueError as e:
            print(f"Skipping item due to error: {e}")
            continue

    # 保存结果
    save_json(train_data, TRAIN_JSON_PATH)
    save_json(test_data, TEST_JSON_PATH)

    print("分割完成。")
    print(f"Train set (ID 0-{SPLIT_THRESHOLD}): {len(train_data)} items -> {TRAIN_JSON_PATH}")
    print(f"Test set (ID >{SPLIT_THRESHOLD}): {len(test_data)} items -> {TEST_JSON_PATH}")


if __name__ == "__main__":
    main()
