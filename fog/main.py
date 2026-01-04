# 给图像添加雾天效果

import os
import cv2
import numpy as np
import random
from tqdm import tqdm


# ============================
# 1. 你的 add_hazy（无需改动）
# ============================
def add_hazy(image, beta=0.05, brightness=0.5):
    img_f = image.astype(np.float32) / 255.0
    row, col, chs = image.shape
    size = np.sqrt(max(row, col))
    center = (row // 2, col // 2)
    y, x = np.ogrid[:row, :col]
    dist = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    d = -0.04 * dist + size
    td = np.exp(-beta * d)
    img_f = img_f * td[..., np.newaxis] + brightness * (1 - td[..., np.newaxis])
    hazy_img = np.clip(img_f * 255, 0, 255).astype(np.uint8)
    return hazy_img


# ============================
# 2. 工具函数
# ============================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def process_height_folder(input_folder, output_folder, beta, brightness):
    """处理一个高度目录（150/300）下所有图片"""
    ensure_dir(output_folder)

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue

        img_path = os.path.join(input_folder, filename)
        image = cv2.imread(img_path)

        if image is None:
            print(f"⚠ 无法读取：{img_path}")
            continue

        foggy = add_hazy(image, beta=beta, brightness=brightness)

        out_path = os.path.join(output_folder, filename)
        cv2.imwrite(out_path, foggy)


# ============================
# 3. 主函数：随机 40 folders，处理 150/300
# ============================
def process_sues200_fog(
        input_root,
        output_root,
        beta=0.05,
        brightness=0.8,
        select_num=40,
        target_heights=("150", "300")
):
    ensure_dir(output_root)

    # 0001 ~ 0200
    all_folders = sorted([f for f in os.listdir(input_root) if f.isdigit()])
    selected = random.sample(all_folders, select_num)

    print("📌 随机选中的 40 个文件夹：")
    print(selected)

    for folder in tqdm(selected, desc="Processing"):
        folder_path = os.path.join(input_root, folder)

        for height in target_heights:
            height_path = os.path.join(folder_path, height)
            if not os.path.exists(height_path):
                continue

            # 保持目录结构
            out_height_path = os.path.join(output_root, folder, height)

            process_height_folder(height_path, out_height_path, beta, brightness)

    print("\n🎉 雾霾数据增强完成！")


# ============================
# 4. 运行入口
# ============================
if __name__ == "__main__":

    # 输入 / 输出根目录
    INPUT_ROOT = r"D:\datasets\SUES-200-512x512-V2\drone_view_512"
    OUTPUT_ROOT = r"D:\datasets\dataAug-SUES-200\\fog"

    beta = 0.05          # 雾强度
    brightness = 0.8     # 雾亮度（越大越亮）

    process_sues200_fog(INPUT_ROOT, OUTPUT_ROOT, beta, brightness)





# import os
# import cv2
# import numpy as np

# def add_hazy(image, beta=0.05, brightness=0.5):
#     """
#     给图像添加雾霾效果
#     :param image: 输入图像
#     :param beta: 雾强
#     :param brightness: 雾霾亮度
#     :return: 添加雾霾后的图像
#     """
#     img_f = image.astype(np.float32) / 255.0
#     row, col, chs = image.shape
#     size = np.sqrt(max(row, col))
#     center = (row // 2, col // 2)
#     y, x = np.ogrid[:row, :col]
#     dist = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
#     d = -0.04 * dist + size
#     td = np.exp(-beta * d)
#     img_f = img_f * td[..., np.newaxis] + brightness * (1 - td[..., np.newaxis])
#     hazy_img = np.clip(img_f * 255, 0, 255).astype(np.uint8)
#     return hazy_img


# def process_folder(input_folder, output_folder, beta=0.05, brightness=0.5):
#     """
#     批量处理输入文件夹中的图像，添加雾霾效果并保存到输出文件夹
#     :param input_folder: 输入图像文件夹
#     :param output_folder: 输出图像文件夹
#     :param beta: 雾强
#     :param brightness: 雾霾亮度
#     """
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     for filename in os.listdir(input_folder):
#         if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
#             input_path = os.path.join(input_folder, filename)
#             image = cv2.imread(input_path)
#             if image is None:
#                 print(f"⚠️ 无法读取图像 {filename}, 跳过")
#                 continue

#             hazy_image = add_hazy(image, beta=beta, brightness=brightness)
#             output_path = os.path.join(output_folder, filename)
#             cv2.imwrite(output_path, hazy_image)
#             print(f"✅ 已生成 -> {output_path}")


# if __name__ == '__main__':
#     input_folder = r'input_img'    # 输入文件夹路径
#     output_folder = r'foggy_output'  # 输出文件夹路径

#     # 调整雾强和亮度
#     beta = 0.05
#     brightness = 0.8

#     process_folder(input_folder, output_folder, beta, brightness)
