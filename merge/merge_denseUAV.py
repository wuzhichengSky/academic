import json
import re
import os

# =========================
# Global config (file names)
# =========================
DRONE_JSON_PATH = "drone.json"
SATELLITE_JSON_PATH = "satellite.json"
OUTPUT_JSON_PATH = "merged_denseUAV.json"


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def get_key_from_drone_path(path: str) -> str:
    """
    从无人机路径提取唯一匹配键。
    输入示例: drone/000000/H80.JPG
    输出示例: 000000/H80
    """
    # 匹配 drone/ 后面的所有内容，直到遇到最后一个点号（去掉后缀）
    # 也可以使用 os.path 处理，这里用正则以保持风格一致
    # 假设路径结构固定为: root/scene_id/filename.ext
    
    # 方法1：正则提取 drone/ 之后的内容，去掉后缀
    m = re.search(r"drone/(.+)\.[^.]+$", path, re.IGNORECASE)
    if m:
        return m.group(1)
    
    # 方法2：如果正则失败，尝试用 os.path 分割
    # 假设结构是 drone/ID/Name.jpg
    try:
        parts = path.replace("\\", "/").split("/")
        # 取最后两个部分：ID 和 文件名
        scene_id = parts[-2]
        filename_stem = os.path.splitext(parts[-1])[0]
        return f"{scene_id}/{filename_stem}"
    except Exception:
        raise ValueError(f"Cannot parse key from drone path: {path}")


def get_key_from_sat_path(path: str) -> str:
    """
    从卫星路径提取唯一匹配键。
    输入示例: satellite/000000/H80.tif (或 H90.tif)
    输出示例: 000000/H80
    """
    m = re.search(r"satellite/(.+)\.[^.]+$", path, re.IGNORECASE)
    if m:
        return m.group(1)
        
    try:
        parts = path.replace("\\", "/").split("/")
        scene_id = parts[-2]
        filename_stem = os.path.splitext(parts[-1])[0]
        return f"{scene_id}/{filename_stem}"
    except Exception:
        raise ValueError(f"Cannot parse key from satellite path: {path}")


def get_first_sentence(text: str) -> str:
    """获取文本的第一句（包含 '.'）"""
    idx = text.find(".")
    if idx == -1:
        return text.strip()
    return text[: idx + 1].strip()


def remove_first_sentence_keep_rest(text: str) -> str:
    """去掉文本的第一句，保留剩余部分"""
    idx = text.find(".")
    if idx == -1:
        return ""
    return text[idx + 1 :].strip()


def lowercase_first_alpha(s: str) -> str:
    """将字符串中第一个字母字符变为小写"""
    for i, ch in enumerate(s):
        if ch.isalpha():
            return s[:i] + ch.lower() + s[i + 1 :]
    return s


def main():
    drone_data = load_json(DRONE_JSON_PATH)
    sat_data = load_json(SATELLITE_JSON_PATH)

    # 1. 构建卫星数据的索引
    # Key 为 "scene_id/filename_stem" (如 000000/H80)
    # Value 为 (完整路径, caption)
    key_to_sat = {}
    for item in sat_data:
        if not isinstance(item, dict) or len(item) != 1:
            print(f"Warning: Skipping invalid format item in satellite.json: {item}")
            continue
            
        sat_path, sat_cap = next(iter(item.items()))
        try:
            key = get_key_from_sat_path(sat_path)
            key_to_sat[key] = (sat_path, sat_cap)
        except ValueError as e:
            print(e)
            continue

    merged = []

    # 2. 遍历无人机数据并进行匹配
    for item in drone_data:
        if not isinstance(item, dict) or len(item) != 1:
            print(f"Warning: Skipping invalid format item in drone.json: {item}")
            continue
            
        drone_path, drone_cap = next(iter(item.items()))
        
        try:
            # 提取键，如 000000/H80
            key = get_key_from_drone_path(drone_path)
        except ValueError as e:
            print(e)
            continue

        # 在卫星数据中查找匹配的键
        if key not in key_to_sat:
            # 如果找不到完全匹配的文件名（如 H80 对应 H80），则跳过
            # 如果您的需求确实是 H80 强行对应 H90，则需要额外的映射表逻辑，
            # 但基于提供的 JSON 样本，H80和H90在两边都存在，因此推测是同名匹配。
            continue

        sat_path, sat_cap = key_to_sat[key]

        # 3. 文本合并逻辑
        # 第一条总结句：沿用无人机第一句
        first_sentence = get_first_sentence(drone_cap)

        # 剩余部分处理
        drone_rest = lowercase_first_alpha(remove_first_sentence_keep_rest(drone_cap))
        sat_rest = lowercase_first_alpha(remove_first_sentence_keep_rest(sat_cap))

        caption = (
            f"{first_sentence}"
            f"From a drone view, {drone_rest}"
            f"From a satellite view, {sat_rest}"
        ).strip()

        merged.append({
            "drone": drone_path,
            "satellite": sat_path,
            "caption": caption
        })

    save_json(merged, OUTPUT_JSON_PATH)
    print(f"Done. Merged items: {len(merged)} -> {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()
