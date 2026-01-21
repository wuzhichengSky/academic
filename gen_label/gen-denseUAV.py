# 通过大模型生成图片描述标签

import requests
import base64
import os
import json
from tqdm import tqdm  # pip install tqdm
import time
import re
from datetime import datetime
from PIL import Image
from io import BytesIO

API_URL = "https://api.siliconflow.cn/v1/chat/completions"
API_KEY = "sk-umxuhqcnmfwkolzkxkkqelrxsrzqjkzneixinluckhdosdrl"

IMAGE_ROOT = r"D:\dataset-w\DenseUAV-text\satellite"
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
OUTPUT_JSON = f"satellite-1501-3033-{timestamp}.json"
ERROR_LOG = "error.log"


def encode_image(image_file):
    ext = os.path.splitext(image_file)[1].lower()

    # tif/tiff -> 转成 JPEG 再编码（最省事、兼容性最好）
    if ext in [".tif", ".tiff"]:
        img = Image.open(image_file).convert("RGB")
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=95)  # 质量可调
        return base64.b64encode(buf.getvalue()).decode("utf-8"), "image/jpeg"

    with open(image_file, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8'), "image/jpeg"


def request_caption(image_path, max_retries=3):
    img_b64, mime = encode_image(image_path)

    # ===== 根据路径判断高度 =====
    image_path_unix = image_path.replace("\\", "/")
    filename = os.path.basename(image_path_unix).lower()

    if "h80" in filename:
        word_limit = "100–150 words"
    else:
        word_limit = "150–200 words"

    payload = {
        "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{img_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": (
                            f"Provide a detailed English description of this image. Requirements: Begin with a brief statement about the overall content of the image, for example: ‘This is an aerial image of an industrial park.’ Then describe the central object in the picture, followed by the objects around it (you may use positional words such as left and right). The description may include a small amount of emotional color, expressed both locally (adjectives describing objects) and globally (the overall atmosphere at the end of the paragraph). Emotional expression should be minimal. Also include a short description of the weather (such as fog, sunlight, or rain). Use simple vocabulary as much as possible, and return only one paragraph with no line breaks. The word count must be strictly limited to {word_limit}. Below are some emotional descriptive phrases for reference: ‘a busy street,’ ‘a dim small restaurant,’ ‘a quiet and cozy street café,’ ‘warm lights.’"
                        )
                    }
                ]
            }
        ]
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    # ===== 发送请求 + 重试 =====
    for i in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=40)
            result = response.json()

            if "choices" in result:
                return result["choices"][0]["message"]["content"]

            print(f"⚠️ 第 {i+1} 次尝试失败（无 choices），重试中…")
            time.sleep(2)

        except Exception as e:
            print(f"⚠️ 请求异常：{e}，第 {i+1} 次重试中…")
            time.sleep(2)

    # ===== 三次失败 → 写入日志 =====
    with open(ERROR_LOG, "a", encoding="utf-8") as log:
        log.write(image_path + "\n")

    return "ERROR"


def natural_key(s):
    # 将路径中的数字提取出来按数值排序，例如 150/10.jpg → ["150", "10"]
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r"(\d+)", s)]

def collect_images_drone(root):
    imgs = []

    # 如果直接传入的是单张图片
    if os.path.isfile(root) and root.lower().endswith(".jpg"):
        return [root.replace("\\", "/")]

    # 遍历 000000 - 003032 场景文件夹
    for i in range(1501, 3033):
        scene_folder = os.path.join(root, f"{i:06d}")
        if not os.path.exists(scene_folder):
            continue

        # 只收集 H80 / H90 / H100
        for name in ["H80.JPG", "H90.JPG", "H100.JPG"]:
            img_path = os.path.join(scene_folder, name)
            if os.path.exists(img_path):
                imgs.append(img_path.replace("\\", "/"))

    # 自然排序
    imgs = sorted(imgs, key=natural_key)

    return imgs

def collect_images_satellite(root):
    imgs = []

    # 如果直接传入的是单张图片
    if os.path.isfile(root) and root.lower().endswith(".tif"):
        return [root.replace("\\", "/")]

    # 遍历 000000 - 003032 场景文件夹
    for i in range(1501, 3033):
        scene_folder = os.path.join(root, f"{i:06d}")
        if not os.path.exists(scene_folder):
            continue

        # 只收集 H80 / H90 / H100
        for name in ["H80.tif", "H90.tif", "H100.tif"]:
            img_path = os.path.join(scene_folder, name)
            if os.path.exists(img_path):
                imgs.append(img_path.replace("\\", "/"))

    # 自然排序
    imgs = sorted(imgs, key=natural_key)

    return imgs


if __name__ == "__main__":
    # ======== 启动时清空错误日志 ========
    with open(ERROR_LOG, "w", encoding="utf-8") as log:
        log.write("")

    images = collect_images_satellite(IMAGE_ROOT)
    label_list = []

    print(f"发现 {len(images)} 张图片 ✅ 开始生成标签...\n")


    for img_path in tqdm(images):
        # 1. 统一斜杠
        img_path = img_path.replace("\\", "/")

        # 2. 保留从 drone_view_512/ 开始的路径
        key_path = img_path.split("satellite/", 1)[-1]
        key_path = "satellite/" + key_path

        caption = request_caption(img_path)

        label_list.append({key_path: caption})



    # ======== 写入 JSON 文件 ========
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(label_list, f, ensure_ascii=False, indent=4)

    # ======== 🔥 执行结束统计错误数量 ========
    with open(ERROR_LOG, "r", encoding="utf-8") as log:
        errors = [line.strip() for line in log.readlines() if line.strip()]

    error_count = len(errors)

    print("\n=======================")
    print("🎉 本次执行已完成")
    print("=======================")

    if error_count == 0:
        print("✅ 本次执行 *没有出现任何错误*")
    else:
        print(f"⚠️ 本次执行出现 **{error_count} 个错误**")
        print("❗ 错误图片路径已记录在 error.log 中")

    print(f"📄 标签文件：{OUTPUT_JSON}")
    print(f"📄 错误日志文件：{ERROR_LOG}")
