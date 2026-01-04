import os
import json
import torch
import clip
import re
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# ============ 配置 ============
device = "cuda" if torch.cuda.is_available() else "cpu"

image_root = r"D:\dataset-w\SUES-200-512x512-V2\SUES-200-512x512"
train_json = "./train.json"
test_json  = "./test.json"

batch_size = 16                     # ✅ 提高 batch
epochs = 20                         # ✅ 增加训练轮数
lr = 1e-5
num_workers = 4

# ============ 文本清洗 ============
def clean_text(s: str) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)

    s = re.sub(r"(?i)\bWord\s*count\s*:\s*\d+\.?$", "", s.strip())
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_prompt(text):
    # ✅ 改进成 CLIP 更友好的 prompt
    return f"aerial image: {text}"

# ============ 数据集 ============
class CLIPDataset(Dataset):
    def __init__(self, json_path, image_root, preprocess):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.image_root = image_root
        self.keys = []
        self.texts = []
        self.preprocess = preprocess

        for item in self.data:
            for k, v in item.items():
                self.keys.append(k)
                self.texts.append(build_prompt(clean_text(v)))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_root, self.keys[idx])
        text = self.texts[idx]

        image = Image.open(img_path).convert("RGB")
        image = self.preprocess(image)
        return image, text

# ============ 训练 ============
def train():
    model, preprocess = clip.load(
        "ViT-B/32",
        device=device,
        download_root="../clip_weights",
        jit=False
    )

    train_dataset = CLIPDataset(train_json, image_root, preprocess)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    print("✅ Start Training")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, texts in tqdm(train_loader):
            images = images.to(device)
            texts = [t if isinstance(t, str) else str(t) for t in texts]
            tokens = clip.tokenize(texts, truncate=True).to(device)

            # ✅ 正确的 CLIP forward
            logits_per_image, logits_per_text = model(images, tokens)

            labels = torch.arange(len(images)).long().to(device)

            loss = (loss_fn(logits_per_image, labels) +
                    loss_fn(logits_per_text, labels)) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "clip_sues.pth")
    print("✅ Model saved")

# ============ 测试 ============
@torch.no_grad()
def test():
    model, preprocess = clip.load(
        "ViT-B/32",
        device=device,
        download_root="../clip_weights",
        jit=False
    )

    model.load_state_dict(torch.load("clip_sues.pth", map_location=device))
    model.eval()

    test_dataset = CLIPDataset(test_json, image_root, preprocess)
    texts = test_dataset.texts

    # 解析 GT scene
    gt_scenes = []
    for k in test_dataset.keys:
        parts = k.replace("\\", "/").split("/")
        gt_scenes.append(parts[1] if len(parts) > 1 else "")

    # 收集无人机图库
    drone_root = os.path.join(image_root, "drone_view_512")
    drone_paths, drone_scenes = [], []
    for r, d, f_list in os.walk(drone_root):
        for f in f_list:
            if f.lower().endswith(".jpg"):
                full = os.path.join(r, f)
                rel = os.path.relpath(full, image_root).replace("\\", "/")
                parts = rel.split("/")
                drone_paths.append(full)
                drone_scenes.append(parts[1] if len(parts) > 1 else "")

    # 收集卫星图库
    satellite_root = os.path.join(image_root, "satellite-view")
    sat_paths, sat_scenes = [], []
    for scene in os.listdir(satellite_root):
        p = os.path.join(satellite_root, scene, "0.png")
        if os.path.exists(p):
            sat_paths.append(p)
            sat_scenes.append(scene)

    # 图像特征提取
    def encode_images(paths):
        feats = []
        for i in tqdm(range(0, len(paths), batch_size)):
            batch = paths[i:i+batch_size]
            imgs = [preprocess(Image.open(p).convert("RGB")) for p in batch]
            imgs = torch.stack(imgs).to(device)

            f = model.encode_image(imgs)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu())
        return torch.cat(feats, dim=0)

    print("✅ Encoding drone images")
    drone_feats = encode_images(drone_paths)

    print("✅ Encoding satellite images")
    sat_feats = encode_images(sat_paths)

    # 文本特征
    text_feats = []
    for i in tqdm(range(0, len(texts), batch_size)):
        bt = texts[i:i+batch_size]
        tokens = clip.tokenize(bt, truncate=True).to(device)
        f = model.encode_text(tokens)
        f = f / f.norm(dim=-1, keepdim=True)
        text_feats.append(f.cpu())
    text_feats = torch.cat(text_feats, dim=0)

    # 相似度
    sims_drone = (text_feats @ drone_feats.t()).numpy()
    sims_sat   = (text_feats @ sat_feats.t()).numpy()

    # 评估
    ks = [1, 5, 10]
    res = {k: {"drone":0, "sat":0, "both":0} for k in ks}
    total = len(texts)

    for i in range(total):
        gt = gt_scenes[i]

        rank_dr = np.argsort(sims_drone[i])[::-1]
        rank_sa = np.argsort(sims_sat[i])[::-1]

        for k in ks:
            pred_dr = [drone_scenes[x] for x in rank_dr[:k]]
            pred_sa = [sat_scenes[x] for x in rank_sa[:k]]

            ok_dr = gt in pred_dr
            ok_sa = gt in pred_sa

            if ok_dr:
                res[k]["drone"] += 1
            if ok_sa:
                res[k]["sat"] += 1
            if ok_dr and ok_sa:
                res[k]["both"] += 1

    print("\n📊 Results:")
    for k in ks:
        print(
            f"Top-{k} | Drone: {res[k]['drone']/total*100:.2f}% "
            f"Satellite: {res[k]['sat']/total*100:.2f}% "
            f"Both: {res[k]['both']/total*100:.2f}%"
        )

# ============ 主入口 ============
if __name__ == "__main__":
    train()
    test()


# import os
# import json
# import torch
# import clip
# import re
# import numpy as np
# from PIL import Image
# from tqdm import tqdm
# from torch.utils.data import Dataset, DataLoader

# # ============ 配置 ============
# device = "cuda" if torch.cuda.is_available() else "cpu"

# image_root = r"D:\dataset-w\SUES-200-512x512-V2\SUES-200-512x512"               # 改成你的图像根目录
# train_json = "./train.json"     # 你的训练标注
# test_json  = "./test.json"      # 你的测试标注

# batch_size = 2
# epochs = 5
# lr = 1e-5
# num_workers = 4

# def truncate_text(t, max_len=70):
#     words = t.split()
#     if len(words) <= max_len:
#         return t
#     return " ".join(words[:max_len])

# # ============ 数据集 ============
# # class CLIPDataset(Dataset):
# #     def __init__(self, json_path, image_root):
# #         with open(json_path, 'r', encoding='utf-8') as f:
# #             self.data = json.load(f)

# #         self.image_root = image_root
# #         self.keys = []
# #         self.texts = []

# #         for item in self.data:
# #             for k, v in item.items():
# #                 self.keys.append(k)
# #                 self.texts.append(v)

# #     def __len__(self):
# #         return len(self.keys)

# #     def __getitem__(self, idx):
# #         img_path = os.path.join(self.image_root, self.keys[idx])
# #         text = self.texts[idx]

# #         image = Image.open(img_path).convert("RGB")
# #         return image, text

# def clean_text(s: str) -> str:
#     """
#     清洗文本：
#     - 去掉 'Word count: 142' 这种尾部计数信息
#     - 合并多空白为单个空格
#     - 去掉开头/结尾空白
#     - 保证是字符串
#     """
#     if s is None:
#         return ""
#     if not isinstance(s, str):
#         s = str(s)

#     # 去掉类似 "Word count: 142." 或 "Word count: 142" 的尾部行
#     s = re.sub(r"(?i)\bWord\s*count\s*:\s*\d+\.?$", "", s.strip())

#     # 去掉其他末尾形式的“Word count”出现在句尾的情况
#     s = re.sub(r"(?i)\s*Word\s*count\s*[:\-]\s*\d+\s*\.?$", "", s)

#     # 合并多空白和换行
#     s = re.sub(r"\s+", " ", s).strip()

#     return s

# class CLIPDataset(Dataset):
#     def __init__(self, json_path, image_root, preprocess):
#         with open(json_path, 'r', encoding='utf-8') as f:
#             self.data = json.load(f)

#         self.image_root = image_root
#         self.keys = []
#         self.texts = []
#         self.preprocess = preprocess

#         for item in self.data:
#             for k, v in item.items():
#                 self.keys.append(k)
#                 # 这里直接清洗并保存文本字符串，避免 worker 问题
#                 self.texts.append(clean_text(v))

#     def __len__(self):
#         return len(self.keys)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.image_root, self.keys[idx])
#         text = self.texts[idx]

#         image = Image.open(img_path).convert("RGB")
#         image = self.preprocess(image)   # 直接返回 tensor

#         return image, text


# # ============ Training ============
# def train():
#     model, preprocess = clip.load(
#         "ViT-B/32",
#         device=device,
#         download_root="../clip_weights",
#         jit=False)

#     # train_dataset = CLIPDataset(train_json, image_root)
#     train_dataset = CLIPDataset(train_json, image_root, preprocess)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     loss_fn = torch.nn.CrossEntropyLoss()

#     print("✅ Start Training")

#     for epoch in range(epochs):
#         model.train()
#         epoch_loss = 0

#         for images, texts in tqdm(train_loader):
#             # images = torch.stack([preprocess(img) for img in images]).to(device)
#             images = images.to(device)
#             # texts = clip.tokenize(texts).to(device)
#             # texts 现在是一个 list[str]，先做额外的防护性清洗（可选）
#             texts = [t if isinstance(t, str) else str(t) for t in texts]

#             # 使用 CLIP 官方提供的 token-level 截断（最稳妥）
#             tokens = clip.tokenize(texts, truncate=True).to(device)

#             image_feat, text_feat = model(images, tokens)

#             logits_per_image = image_feat @ text_feat.t()
#             logits_per_text = logits_per_image.t()

#             labels = torch.arange(len(images)).to(device)

#             loss = (loss_fn(logits_per_image, labels) +
#                     loss_fn(logits_per_text, labels)) / 2

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             epoch_loss += loss.item()

#         print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}")

#     torch.save(model.state_dict(), "clip_sues.pth")
#     print("✅ Model saved as clip_sues.pth")


# # ============ Evaluation ============
# # @torch.no_grad()
# # def test():
# #     model, preprocess = clip.load(
# #         "ViT-B/32",
# #         device=device,
# #         download_root="./clip_weights",
# #         jit=False
# #     )
# #     model.load_state_dict(torch.load("clip_sues.pth", map_location=device))
# #     model.eval()

# #     # ================= 1. 加载文本数据 =================
# #     test_dataset = CLIPDataset(test_json, image_root)
# #     texts = test_dataset.texts

# #     # ================= 2. 构建无人机图库 =================
# #     drone_root = os.path.join(image_root, "drone_view_512")
# #     satellite_root = os.path.join(image_root, "satellite-view")

# #     drone_paths = []
# #     drone_scenes = []

# #     for scene in os.listdir(drone_root):
# #         scene_dir = os.path.join(drone_root, scene)
# #         if not os.path.isdir(scene_dir):
# #             continue

# #         # 找任意一张无人机图作为代表
# #         for root, dirs, files in os.walk(scene_dir):
# #             for f in files:
# #                 if f.endswith(".jpg"):
# #                     full_path = os.path.join(root, f)
# #                     drone_paths.append(full_path)
# #                     drone_scenes.append(scene)
# #                     break
# #             if len(drone_paths) > 0 and drone_scenes[-1] == scene:
# #                 break

# #     # ================= 3. 构建卫星图库 =================
# #     satellite_paths = []
# #     satellite_scenes = []

# #     for scene in os.listdir(satellite_root):
# #         scene_dir = os.path.join(satellite_root, scene)
# #         if not os.path.isdir(scene_dir):
# #             continue

# #         img_path = os.path.join(scene_dir, f"{scene}.jpg")
# #         if os.path.exists(img_path):
# #             satellite_paths.append(img_path)
# #             satellite_scenes.append(scene)

# #     # ================= 4. 提取无人机特征 =================
# #     print("✅ Extracting drone image features...")
# #     drone_feats = []

# #     for p in tqdm(drone_paths):
# #         img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
# #         feat = model.encode_image(img)
# #         feat = feat / feat.norm(dim=-1, keepdim=True)
# #         drone_feats.append(feat.cpu())

# #     drone_feats = torch.cat(drone_feats, dim=0)

# #     # ================= 5. 提取卫星特征 =================
# #     print("✅ Extracting satellite image features...")
# #     satellite_feats = []

# #     for p in tqdm(satellite_paths):
# #         img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
# #         feat = model.encode_image(img)
# #         feat = feat / feat.norm(dim=-1, keepdim=True)
# #         satellite_feats.append(feat.cpu())

# #     satellite_feats = torch.cat(satellite_feats, dim=0)

# #     # ================= 6. 提取文本特征 =================
# #     print("✅ Extracting text features...")
# #     text_feats = []

# #     for t in tqdm(texts):
# #         token = clip.tokenize(t).to(device)
# #         feat = model.encode_text(token)
# #         feat = feat / feat.norm(dim=-1, keepdim=True)
# #         text_feats.append(feat.cpu())

# #     text_feats = torch.cat(text_feats, dim=0)

# #     # ================= 7. 相似度计算 =================
# #     sims_text_drone = text_feats @ drone_feats.t()
# #     sims_text_sat   = text_feats @ satellite_feats.t()

# #     sims_text_drone = sims_text_drone.numpy()
# #     sims_text_sat   = sims_text_sat.numpy()

# #     # ================= 8. 评估逻辑 =================
# #     success = 0
# #     total = len(texts)

# #     for i in range(total):
# #         # 文本真实 scene（来自原始标注路径）
# #         gt_path = test_dataset.keys[i]
# #         gt_scene = gt_path.split("/")[1]   # drone_view_512/0001/...

# #         # 无人机检索
# #         best_drone_idx = np.argmax(sims_text_drone[i])
# #         pred_drone_scene = drone_scenes[best_drone_idx]

# #         # 卫星检索
# #         best_sat_idx = np.argmax(sims_text_sat[i])
# #         pred_sat_scene = satellite_scenes[best_sat_idx]

# #         if pred_drone_scene == gt_scene and pred_sat_scene == gt_scene:
# #             success += 1

# #     print("\n📊 Cross-view Retrieval Result:")
# #     print(f"Total samples: {total}")
# #     print(f"Both-view correct: {success}")
# #     print(f"Accuracy: {success/total*100:.2f}%")
# @torch.no_grad()
# def test():
#     model, preprocess = clip.load(
#         "ViT-B/32",
#         device=device,
#         download_root="../clip_weights",
#         jit=False
#     )
#     model.load_state_dict(torch.load("clip_sues.pth", map_location=device))
#     model.eval()

#     # ================= 1. 加载测试数据文本和GT scene =================
#     test_dataset = CLIPDataset(test_json, image_root,preprocess)
#     texts = test_dataset.texts
#     gt_scenes = []
#     # 从数据里的 key 推 scene（假设 key 格式为 "drone_view_512/0001/150/1.jpg"）
#     for k in test_dataset.keys:
#         parts = k.split("/")
#         if len(parts) >= 2:
#             gt_scenes.append(parts[1])
#         else:
#             gt_scenes.append("")  # 容错

#     # ================= 2. 收集无人机图库（全部图片） =================
#     drone_root = os.path.join(image_root, "drone_view_512")
#     drone_paths = []
#     drone_scenes = []
#     for root_dir, dirs, files in os.walk(drone_root):
#         for f in files:
#             if f.lower().endswith(".jpg"):
#                 full_path = os.path.join(root_dir, f)
#                 # 相对 image_root 的路径形式
#                 rel = os.path.relpath(full_path, image_root)
#                 parts = rel.replace("\\", "/").split("/")  # 兼容 Windows 路径分隔
#                 # 期望 parts = ['drone_view_512','0001','150','1.jpg']
#                 if len(parts) >= 2:
#                     scene = parts[1]
#                 else:
#                     scene = ""
#                 drone_paths.append(full_path)
#                 drone_scenes.append(scene)

#     # ================= 3. 收集卫星图库（每场景一张卫星图，格式：scene/scene.jpg） =================
#     satellite_root = os.path.join(image_root, "satellite-view")
#     satellite_paths = []
#     satellite_scenes = []
#     for scene in os.listdir(satellite_root):
#         scene_dir = os.path.join(satellite_root, scene)
#         if not os.path.isdir(scene_dir):
#             continue
#         img_path = os.path.join(scene_dir, "0.png")
#         if os.path.exists(img_path):
#             satellite_paths.append(img_path)
#             satellite_scenes.append(scene)

#     # ================ 4. 提取无人机图像特征（按 batch 提取，避免逐张过慢） ================
#     print("✅ Extracting drone image features...")
#     drone_feats_list = []
#     bs_img = batch_size  # 重用训练的 batch_size
#     for i in tqdm(range(0, len(drone_paths), bs_img)):
#         batch_paths = drone_paths[i: i + bs_img]
#         imgs = []
#         for p in batch_paths:
#             img = Image.open(p).convert("RGB")
#             imgs.append(preprocess(img))
#         imgs = torch.stack(imgs).to(device)
#         feats = model.encode_image(imgs)
#         feats = feats / feats.norm(dim=-1, keepdim=True)
#         drone_feats_list.append(feats.cpu())
#     if len(drone_feats_list) == 0:
#         raise RuntimeError("No drone images found in: " + drone_root)
#     drone_feats = torch.cat(drone_feats_list, dim=0)  # [N_drone, D]

#     # ================ 5. 提取卫星图像特征（按 batch） ================
#     print("✅ Extracting satellite image features...")
#     sat_feats_list = []
#     for i in tqdm(range(0, len(satellite_paths), bs_img)):
#         batch_paths = satellite_paths[i: i + bs_img]
#         imgs = []
#         for p in batch_paths:
#             img = Image.open(p).convert("RGB")
#             imgs.append(preprocess(img))
#         imgs = torch.stack(imgs).to(device)
#         feats = model.encode_image(imgs)
#         feats = feats / feats.norm(dim=-1, keepdim=True)
#         sat_feats_list.append(feats.cpu())
#     if len(sat_feats_list) == 0:
#         raise RuntimeError("No satellite images found in: " + satellite_root)
#     satellite_feats = torch.cat(sat_feats_list, dim=0)  # [N_sat, D]

#     # ================ 6. 提取文本特征（按 batch） ================
#     print("✅ Extracting text features...")
#     text_feats_list = []
#     for i in tqdm(range(0, len(texts), bs_img)):
#         batch_texts = texts[i: i + bs_img]
#         tokens = clip.tokenize(batch_texts, truncate=True).to(device)  # returns tensor [B, L]
#         feats = model.encode_text(tokens)
#         feats = feats / feats.norm(dim=-1, keepdim=True)
#         text_feats_list.append(feats.cpu())
#     text_feats = torch.cat(text_feats_list, dim=0)  # [N_text, D]

#     # ================ 7. 计算相似度矩阵（文本 vs drone），（文本 vs satellite） ================
#     print("✅ Computing similarity matrices...")
#     sims_text_drone = (text_feats @ drone_feats.t()).numpy()      # [N_text, N_drone]
#     sims_text_sat   = (text_feats @ satellite_feats.t()).numpy()  # [N_text, N_sat]

#     # ================ 8. 评估 Top-1/5/10（同时统计单视角和两视角同时成功） ================
#     ks = [1, 5, 10]
#     drone_counts = {k: 0 for k in ks}
#     sat_counts = {k: 0 for k in ks}
#     combined_counts = {k: 0 for k in ks}
#     total = len(texts)

#     for i in range(total):
#         gt_scene = gt_scenes[i]

#         # drone top-k
#         ranks_drone = np.argsort(sims_text_drone[i])[::-1]  # 大到小
#         # satellite top-k
#         ranks_sat = np.argsort(sims_text_sat[i])[::-1]

#         for k in ks:
#             topk_dr = ranks_drone[:k]
#             topk_sat = ranks_sat[:k]

#             # check if any of topk_dr scenes equals gt_scene
#             pred_drone_scenes = [drone_scenes[idx] for idx in topk_dr]
#             pred_sat_scenes = [satellite_scenes[idx] for idx in topk_sat]

#             ok_drone = gt_scene in pred_drone_scenes
#             ok_sat = gt_scene in pred_sat_scenes

#             if ok_drone:
#                 drone_counts[k] += 1
#             if ok_sat:
#                 sat_counts[k] += 1
#             if ok_drone and ok_sat:
#                 combined_counts[k] += 1

#     # ================ 9. 输出结果 ================
#     print("\n📊 Retrieval Results (scene-level):")
#     for k in ks:
#         d_acc = drone_counts[k] / total * 100.0
#         s_acc = sat_counts[k] / total * 100.0
#         c_acc = combined_counts[k] / total * 100.0
#         print(f"Top-{k}: Drone acc = {d_acc:.2f}%, Satellite acc = {s_acc:.2f}%, Both acc = {c_acc:.2f}%")

#     # 也返回详细数值以便后续处理
#     return {
#         "drone_counts": drone_counts,
#         "sat_counts": sat_counts,
#         "combined_counts": combined_counts,
#         "total": total,
#         "drone_paths": drone_paths,
#         "drone_scenes": drone_scenes,
#         "satellite_paths": satellite_paths,
#         "satellite_scenes": satellite_scenes
#     }


# # ============ Main ============
# if __name__ == "__main__":
#     #train()
#     test()
