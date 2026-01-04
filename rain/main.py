# 给图片添加雨天效果

import cv2
import albumentations as A
import os
import random
from tqdm import tqdm

# =========================
# 1. 输入输出目录（按你的 SUES-200 结构）
# =========================
INPUT_ROOT = r"D:\datasets\SUES-200-512x512-V2\drone_view_512"
OUTPUT_ROOT = r"D:\datasets\dataAug-SUES-200\\rain"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# 随机选择的 folder 数量
SELECT_NUM = 40

# 只处理的高度文件夹
TARGET_HEIGHTS = ["150", "300"]


# =========================
# 2. 雨天增强 transform（你的原代码）
# =========================
transform = A.Compose([

    # 第一层细雨
    A.RandomRain(
        slant_lower=-10,
        slant_upper=10,
        drop_length=25,
        drop_width=1,
        drop_color=(180,180,180),
        blur_value=3,
        rain_type="drizzle",
        p=1.0
    ),

    # 第二层中等雨
    A.RandomRain(
        slant_lower=-15,
        slant_upper=15,
        drop_length=35,
        drop_width=1,
        drop_color=(170, 170, 170),
        blur_value=5,
        rain_type="drizzle",
        p=1.0
    ),

    # 第三层更长更密的细雨
    A.RandomRain(
        slant_lower=-20,
        slant_upper=20,
        drop_length=45,
        drop_width=1,
        drop_color=(150, 150, 150),
        blur_value=7,
        rain_type="drizzle",
        p=1.0
    ),

    # 环境亮度调整
    A.RandomBrightnessContrast(
        brightness_limit=(-0.03, -0.02),
        contrast_limit=(-0.05, 0.04),
        p=1.0
    ),
])


# =========================
# 工具函数
# =========================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def process_folder(input_folder, output_folder):
    """处理一个高度目录（150 或 300）下所有图片"""

    ensure_dir(output_folder)

    for img_name in os.listdir(input_folder):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        in_path = os.path.join(input_folder, img_name)
        img = cv2.imread(in_path)

        if img is None:
            print(f"❌ 无法读取：{in_path}")
            continue

        aug_img = transform(image=img)["image"]

        out_path = os.path.join(output_folder, img_name)
        cv2.imwrite(out_path, aug_img)


# =========================
# 3. 主流程
# =========================
def main():
    # 0001 ~ 0200
    all_folders = sorted([f for f in os.listdir(INPUT_ROOT) if f.isdigit()])

    # 随机选取 40 个
    selected = random.sample(all_folders, SELECT_NUM)
    print("随机选取的 40 个文件夹：", selected)

    for folder in tqdm(selected, desc="处理中..."):
        folder_path = os.path.join(INPUT_ROOT, folder)

        for height in TARGET_HEIGHTS:
            height_path = os.path.join(folder_path, height)

            if not os.path.exists(height_path):
                continue

            # 输出路径保持相同结构
            out_folder = os.path.join(OUTPUT_ROOT, folder, height)

            process_folder(height_path, out_folder)

    print("\n🎉 所有图片已处理完毕！")


if __name__ == "__main__":
    main()


# import cv2
# import albumentations as A
# import os

# # =========================
# # 1. 配置输入输出文件夹路径
# # =========================
# input_dir = "./input_img"                     # 原始图片所在文件夹
# output_dir = "./rainy_output"       # 加雨后图片输出文件夹
# os.makedirs(output_dir, exist_ok=True)

# # =========================
# # 2. 雨天数据增强 transform
# # =========================
# transform = A.Compose([

#     # -------------------------------------------------
#     # 第一层：基础细雨（最轻、最均匀，用于整体氛围）
#     # -------------------------------------------------
#     A.RandomRain(
#         slant_lower=-10,
#         slant_upper=10,
#         drop_length=25,
#         drop_width=1,                    # 细雨 —— 细
#         drop_color=(180,180,180),      # 较亮，轻微存在感
#         blur_value=3,
#         rain_type="drizzle",
#         p=1.0
#     ),

#     # -------------------------------------------------
#     # 第二层：中等密度雨（拉长+略粗增加真实感）
#     # -------------------------------------------------
#     A.RandomRain(
#         slant_lower=-15,
#         slant_upper=15,
#         drop_length=35,
#         drop_width=1,                    # 中等粗细
#         drop_color=(170, 170, 170),      # 更暗一些，层次感更强
#         blur_value=5,
#         rain_type="drizzle",
#         p=1.0
#     ),

#     # -------------------------------------------------
#     # 第三层：高密度超长细雨（更密、更长、更细）
#     # 通过降低亮度+增加模糊，让雨线视觉“更细”
#     # -------------------------------------------------
#     A.RandomRain(
#         slant_lower=-20,
#         slant_upper=20,
#         drop_length=45,                  # 最长的雨滴
#         drop_width=1,                    # Albumentations 最细=1，但我们让它更“视觉细”
#         drop_color=(150, 150, 150),      # 更暗 → 看起来更细、更轻
#         blur_value=7,                    # 更强模糊 → 视觉更细丝感
#         rain_type="drizzle",
#         p=1.0
#     ),

#     # -------------------------------------------------
#     # 环境调整：轻微变暗但整体偏亮（已调好）
#     # -------------------------------------------------
#     A.RandomBrightnessContrast(
#         brightness_limit=(-0.03, -0.02),
#         contrast_limit=(-0.03, 0.04),
#         p=1.0
#     ),
# ])


# # =========================
# # 3. 遍历文件夹并处理图片
# # =========================
# for filename in os.listdir(input_dir):
#     if filename.lower().endswith((".jpg", ".png", ".jpeg")):
#         img_path = os.path.join(input_dir, filename)
#         img = cv2.imread(img_path)

#         if img is None:
#             print(f"❌ 无法读取：{filename}")
#             continue

#         aug_img = transform(image=img)["image"]

#         out_path = os.path.join(output_dir, f"rainy_{filename}")
#         cv2.imwrite(out_path, aug_img)
#         print(f"✅ 已生成：{out_path}")

# print("🎉 所有图片已处理完毕！")