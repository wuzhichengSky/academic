import json
import re

# =========================
# Global config (file names)
# =========================
DRONE_JSON_PATH = "drone.json"
SATELLITE_JSON_PATH = "satellite.json"
OUTPUT_JSON_PATH = "merged.json"


def load_json(path: str):
    """
    读取 JSON 文件并返回 Python 对象（通常是 list/dict）。

    参数:
        path: JSON 文件路径

    返回:
        解析后的 Python 对象（由 json.load 决定）
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str):
    """
    将 Python 对象保存为 JSON 文件。

    参数:
        obj: 需要保存的数据（list/dict等）
        path: 输出 JSON 文件路径

    说明:
        ensure_ascii=False 保证中文等非 ASCII 字符不被转义
        indent=2 让输出更易读
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def get_scene_id_from_drone_path(p: str) -> str:
    """
    从无人机图像路径中解析“场景ID”（四位数字，如 0001）。

    期望输入格式示例:
        drone_view_512/0001/150/0.jpg

    返回:
        scene_id，例如 "0001"
    """
    m = re.search(r"drone_view_512/(\d{4})/", p)
    if not m:
        raise ValueError(f"Cannot parse scene id from drone path: {p}")
    return m.group(1)


def get_scene_id_from_sat_path(p: str) -> str:
    """
    从卫星图像路径中解析“场景ID”（四位数字，如 0001）。

    期望输入格式示例:
        satellite-view/0001/0.png

    返回:
        scene_id，例如 "0001"
    """
    m = re.search(r"satellite-view/(\d{4})/", p)
    if not m:
        raise ValueError(f"Cannot parse scene id from satellite path: {p}")
    return m.group(1)


def get_first_sentence(text: str) -> str:
    """
    获取文本的“第一句”（以第一个 '.' 为句子结束符，包含该 '.'）。

    返回:
        第一条句子字符串（strip 后）
        如果找不到 '.'，则返回全文 strip 的结果
    """
    idx = text.find(".")
    if idx == -1:
        return text.strip()
    return text[: idx + 1].strip()


def remove_first_sentence_keep_rest(text: str) -> str:
    """
    去掉文本的第一句（以第一个 '.' 为界），保留剩余部分。

    返回:
        去掉第一句后的剩余文本（strip 后）
        若没有 '.'，返回空串
    """
    idx = text.find(".")
    if idx == -1:
        return ""
    return text[idx + 1 :].strip()


def lowercase_first_alpha(s: str) -> str:
    """
    将字符串中“第一个字母字符”的大小写变为小写，用于：
    让 'From a drone/satellite view, ...' 后面拼接的第一词首字母为小写。

    规则:
        - 从头遍历，跳过空格、引号、括号等非字母字符
        - 找到第一个 isalpha() 为 True 的字符，将其 lower()
        - 若找不到字母，则原样返回
    """
    for i, ch in enumerate(s):
        if ch.isalpha():
            return s[:i] + ch.lower() + s[i + 1 :]
    return s


def main():
    """
    主流程：
    1) 读取 drone/satellite 两个 JSON
    2) 建索引 scene_id -> (sat_path, sat_caption)
    3) 遍历每条 drone，按 scene_id 找到对应 sat，生成 caption 并输出
    """
    drone_data = load_json(DRONE_JSON_PATH)
    sat_data = load_json(SATELLITE_JSON_PATH)

    # scene_id -> (sat_path, sat_caption)；同scene多条卫星图时保留第一条（最简单）
    scene_to_sat = {}
    for item in sat_data:
        if not isinstance(item, dict) or len(item) != 1:
            raise ValueError("satellite.json format error: each item must be a dict with exactly 1 key.")
        sat_path, sat_cap = next(iter(item.items()))
        scene_id = get_scene_id_from_sat_path(sat_path)
        scene_to_sat.setdefault(scene_id, (sat_path, sat_cap))

    merged = []
    for item in drone_data:
        if not isinstance(item, dict) or len(item) != 1:
            raise ValueError("drone.json format error: each item must be a dict with exactly 1 key.")
        drone_path, drone_cap = next(iter(item.items()))
        scene_id = get_scene_id_from_drone_path(drone_path)

        # 没有对应卫星图就跳过（可按需改成报错）
        if scene_id not in scene_to_sat:
            continue

        sat_path, sat_cap = scene_to_sat[scene_id]

        # 第一条总结句：沿用无人机第一句
        first_sentence = get_first_sentence(drone_cap)

        # 剩余部分：去掉第一句 + 首字母小写（跳过引号等非字母字符）
        drone_rest = lowercase_first_alpha(remove_first_sentence_keep_rest(drone_cap))
        sat_rest = lowercase_first_alpha(remove_first_sentence_keep_rest(sat_cap))

        # 注意：你要求总结句子后面紧跟下一句，没有空格
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
    print(f"Done. merged items: {len(merged)} -> {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()