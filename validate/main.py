import os
import json
import torch
import clip
from torchvision import transforms
import re
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# ============ 配置 ============
device = "cuda" if torch.cuda.is_available() else "cpu"

image_root = r"D:\datasets\SUES-200-512x512-V2"
train_json = "./train.json"
test_json  = "./test.json"

batch_size = 16
epochs = 20

# 方案A：投影层可用稍大 lr，但 logit_scale 必须更小（param groups 单独设）
proj_lr = 5e-5
logit_scale_lr = 1e-6

weight_decay = 0.01
num_workers = 4

NORM_EPS = 1e-6
LOGIT_SCALE_MAX = float(np.log(100.0))  # ~4.6052

# ================= 测试随机性 transform（保持你原本的逻辑） =================
test_transform = transforms.Compose([
    transforms.Resize(224, interpolation=Image.BICUBIC),
    transforms.RandomResizedCrop(
        224,
        scale=(0.9, 1.0),
        ratio=(0.95, 1.05)
    ),
    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.1,
        hue=0.05
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
])

def clean_text(s: str) -> str:
    if s is None or str(s).strip() == "":
        return "empty description"
    s = str(s)
    s = re.sub(r"(?i)\bWord\s*count\s*:\s*\d+\.?$", "", s.strip())
    s = re.sub(r"\s+", " ", s).strip()
    return s

def parse_scene_from_path(rel_path: str) -> str:
    rel_path = rel_path.replace("\\", "/")
    parts = rel_path.split("/")
    # e.g. drone_view_512/0160/300/48.jpg -> scene=0160
    if len(parts) >= 2:
        return parts[1]
    return ""

def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def is_finite_tensor(x: torch.Tensor) -> bool:
    return torch.isfinite(x).all().item()

class CLIPPairedDataset(Dataset):
    """
    新标注格式（list of dict）：
    [
      {"drone": "...", "satellite": "...", "caption": "..."},
      ...
    ]
    """
    def __init__(self, json_path, image_root, image_transform):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.image_root = image_root
        self.image_transform = image_transform

        self.drone_paths = []
        self.sat_paths = []
        self.texts = []
        self.gt_scenes = []

        for item in self.data:
            drone_rel = item["drone"]
            sat_rel   = item["satellite"]
            cap       = clean_text(item.get("caption", ""))

            self.drone_paths.append(drone_rel)
            self.sat_paths.append(sat_rel)
            self.texts.append(cap)
            self.gt_scenes.append(parse_scene_from_path(drone_rel))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        drone_rel = self.drone_paths[idx]
        sat_rel   = self.sat_paths[idx]
        text      = self.texts[idx]

        drone_full = os.path.join(self.image_root, drone_rel)
        sat_full   = os.path.join(self.image_root, sat_rel)

        drone_img = Image.open(drone_full).convert("RGB")
        sat_img   = Image.open(sat_full).convert("RGB")

        drone_img = self.image_transform(drone_img)
        sat_img   = self.image_transform(sat_img)

        return drone_img, sat_img, text


def configure_trainable_params_A(model):
    """
    方案 A：冻结所有，只训练投影层 + logit_scale
    """
    for p in model.parameters():
        p.requires_grad = False

    # logit_scale
    if hasattr(model, "logit_scale") and model.logit_scale is not None:
        model.logit_scale.requires_grad = True
    else:
        raise RuntimeError("model.logit_scale not found")

    # text projection
    if hasattr(model, "text_projection") and model.text_projection is not None:
        model.text_projection.requires_grad = True
    else:
        raise RuntimeError("model.text_projection not found")

    # visual projection (ViT usually has visual.proj)
    if hasattr(model, "visual") and hasattr(model.visual, "proj") and model.visual.proj is not None:
        model.visual.proj.requires_grad = True
    else:
        raise RuntimeError("model.visual.proj not found")

    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    print("✅ Trainable parameters (scheme A):")
    for n, p in trainable:
        print(f"  - {n}: shape={tuple(p.shape)}")
    return {n: p for n, p in trainable}


# ============ Training ============
def train():
    # 1. 加载模型
    model, preprocess = clip.load(
        "ViT-B/32",
        device=device,
        download_root="../../clip_weights",
        jit=False
    )
    
    # 【关键修改】将模型转换为 float32，防止 fp16 训练时的梯度溢出导致 NaN
    if device == "cuda":
        model = model.float()

    # 2. 配置可训练参数
    trainable = configure_trainable_params_A(model)

    train_dataset = CLIPPairedDataset(train_json, image_root, preprocess)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True
    )

    # 3. 优化器配置
    optimizer = torch.optim.AdamW([
        {"params": [trainable["logit_scale"]], "lr": logit_scale_lr, "weight_decay": 0.0},
        {"params": [trainable["text_projection"]], "lr": proj_lr, "weight_decay": weight_decay},
        {"params": [trainable["visual.proj"]], "lr": proj_lr, "weight_decay": weight_decay},
    ])

    loss_fn = torch.nn.CrossEntropyLoss()

    # 初始 clamp
    with torch.no_grad():
        model.logit_scale.clamp_(0, LOGIT_SCALE_MAX)

    print("✅ Start Training (paired: text + drone + satellite) - Force Float32 Mode")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        valid_steps = 0
        skipped_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for step, (drone_imgs, sat_imgs, texts) in enumerate(pbar):
            # 数据搬运
            drone_imgs = drone_imgs.to(device, non_blocking=True)
            sat_imgs   = sat_imgs.to(device, non_blocking=True)
            texts = [t if isinstance(t, str) else str(t) for t in texts]
            tokens = clip.tokenize(texts, truncate=True).to(device, non_blocking=True)

            # ---- encode ----
            # 注意：因为模型已经是 float32，这里计算出来的 feats 也是 float32，数值更稳定
            drone_feats = model.encode_image(drone_imgs)
            sat_feats   = model.encode_image(sat_imgs)
            text_feats  = model.encode_text(tokens)

            # normalize
            drone_feats = l2norm(drone_feats, NORM_EPS)
            sat_feats   = l2norm(sat_feats, NORM_EPS)
            text_feats  = l2norm(text_feats, NORM_EPS)

            # 安全检查（如果转了 float32，这里基本不会再触发了）
            if (not is_finite_tensor(drone_feats)) or (not is_finite_tensor(sat_feats)) or (not is_finite_tensor(text_feats)):
                skipped_steps += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            # logit_scale clamp
            with torch.no_grad():
                model.logit_scale.clamp_(0, LOGIT_SCALE_MAX)
            logit_scale = model.logit_scale.exp()

            # Logits Calculation
            logits_t_d = logit_scale * (text_feats @ drone_feats.t())
            logits_d_t = logits_t_d.t()
            logits_t_s = logit_scale * (text_feats @ sat_feats.t())
            logits_s_t = logits_t_s.t()

            labels = torch.arange(drone_imgs.size(0), device=device)

            loss_drone = (loss_fn(logits_t_d, labels) + loss_fn(logits_d_t, labels)) / 2
            loss_sat   = (loss_fn(logits_t_s, labels) + loss_fn(logits_s_t, labels)) / 2
            loss = (loss_drone + loss_sat) / 2

            # Check Loss NaN
            if not torch.isfinite(loss).item():
                skipped_steps += 1
                optimizer.zero_grad(set_to_none=True)
                # print("Warning: NaN loss detected, skipping step.") # 调试时可以打开
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0
            )

            optimizer.step()

            # 后处理 clamp
            with torch.no_grad():
                model.logit_scale.clamp_(0, LOGIT_SCALE_MAX)

            epoch_loss += loss.item()
            valid_steps += 1

            if epoch == 0 and step < 3:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "exp(scale)": f"{model.logit_scale.exp().item():.2f}"
                })
            else:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # 防止除以零
        if valid_steps > 0:
            avg_loss = epoch_loss / valid_steps
        else:
            avg_loss = float("nan")
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, valid_steps={valid_steps}, skipped_steps={skipped_steps}")

    torch.save(model.state_dict(), "clip_sues.pth")
    print("✅ Model saved as clip_sues.pth")

@torch.no_grad()
def test():
    model, preprocess = clip.load(
        "ViT-B/32",
        device=device,
        download_root="../../clip_weights",
        jit=False
    )
    model.load_state_dict(torch.load("clip_sues.pth", map_location=device))
    model.eval()

    # ================= 1. 加载测试数据文本和GT scene（paired 标注） =================
    test_dataset = CLIPPairedDataset(test_json, image_root, test_transform)
    texts = test_dataset.texts
    gt_scenes = test_dataset.gt_scenes

    # ================= 2. 收集无人机图库（全部图片） =================
    drone_root = os.path.join(image_root, "drone_view_512")
    drone_paths = []
    drone_scenes = []
    for root_dir, dirs, files in os.walk(drone_root):
        for f in files:
            if f.lower().endswith(".jpg"):
                full_path = os.path.join(root_dir, f)
                rel = os.path.relpath(full_path, image_root).replace("\\", "/")
                drone_paths.append(full_path)
                drone_scenes.append(parse_scene_from_path(rel))

    # ================= 3. 收集卫星图库（每场景一张卫星图，scene/0.png） =================
    satellite_root = os.path.join(image_root, "satellite-view")
    satellite_paths = []
    satellite_scenes = []
    for scene in os.listdir(satellite_root):
        scene_dir = os.path.join(satellite_root, scene)
        if not os.path.isdir(scene_dir):
            continue
        img_path = os.path.join(scene_dir, "0.png")
        if os.path.exists(img_path):
            satellite_paths.append(img_path)
            satellite_scenes.append(scene)

    # ================ 4. 提取无人机图像特征（batch） ================
    print("✅ Extracting drone image features...")
    drone_feats_list = []
    bs_img = batch_size
    for i in tqdm(range(0, len(drone_paths), bs_img)):
        batch_paths = drone_paths[i: i + bs_img]
        imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            img = test_transform(img)
            imgs.append(img)
        imgs = torch.stack(imgs).to(device)
        feats = model.encode_image(imgs)
        feats = l2norm(feats, NORM_EPS)
        drone_feats_list.append(feats.cpu())
    if len(drone_feats_list) == 0:
        raise RuntimeError("No drone images found in: " + drone_root)
    drone_feats = torch.cat(drone_feats_list, dim=0)

    # ================ 5. 提取卫星图像特征（batch） ================
    print("✅ Extracting satellite image features...")
    sat_feats_list = []
    for i in tqdm(range(0, len(satellite_paths), bs_img)):
        batch_paths = satellite_paths[i: i + bs_img]
        imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            img = test_transform(img)
            imgs.append(img)
        imgs = torch.stack(imgs).to(device)
        feats = model.encode_image(imgs)
        feats = l2norm(feats, NORM_EPS)
        sat_feats_list.append(feats.cpu())
    if len(sat_feats_list) == 0:
        raise RuntimeError("No satellite images found in: " + satellite_root)
    satellite_feats = torch.cat(sat_feats_list, dim=0)

    # ================ 6. 提取文本特征（batch） ================
    print("✅ Extracting text features...")
    text_feats_list = []
    for i in tqdm(range(0, len(texts), bs_img)):
        batch_texts = texts[i: i + bs_img]
        tokens = clip.tokenize(batch_texts, truncate=True).to(device)
        feats = model.encode_text(tokens)
        feats = l2norm(feats, NORM_EPS)
        text_feats_list.append(feats.cpu())
    text_feats = torch.cat(text_feats_list, dim=0)

    # ================ 7. 计算相似度矩阵（GPU上算，再转CPU） ================
    print("✅ Computing similarity matrices...")
    text_feats_gpu = text_feats.to(device)
    drone_feats_gpu = drone_feats.to(device)
    sat_feats_gpu = satellite_feats.to(device)

    sims_text_drone = (text_feats_gpu @ drone_feats_gpu.t()).cpu().numpy()
    sims_text_sat   = (text_feats_gpu @ sat_feats_gpu.t()).cpu().numpy()

    # ================ 8. 评估 Top-1/5/10（scene-level） ================
    ks = [1, 5, 10]
    drone_counts = {k: 0 for k in ks}
    sat_counts = {k: 0 for k in ks}
    combined_counts = {k: 0 for k in ks}
    total = len(texts)

    for i in range(total):
        gt_scene = gt_scenes[i]

        ranks_drone = np.argsort(sims_text_drone[i])[::-1]
        ranks_sat   = np.argsort(sims_text_sat[i])[::-1]

        for k in ks:
            topk_dr = ranks_drone[:k]
            topk_sat = ranks_sat[:k]

            pred_drone_scenes = [drone_scenes[idx] for idx in topk_dr]
            pred_sat_scenes = [satellite_scenes[idx] for idx in topk_sat]

            ok_drone = gt_scene in pred_drone_scenes
            ok_sat   = gt_scene in pred_sat_scenes

            if ok_drone:
                drone_counts[k] += 1
            if ok_sat:
                sat_counts[k] += 1
            if ok_drone and ok_sat:
                combined_counts[k] += 1

    # ================ 9. 输出结果 ================
    print("\n📊 Retrieval Results (scene-level):")
    for k in ks:
        d_acc = drone_counts[k] / total * 100.0
        s_acc = sat_counts[k] / total * 100.0
        c_acc = combined_counts[k] / total * 100.0
        print(f"Top-{k}: Drone acc = {d_acc:.2f}%, Satellite acc = {s_acc:.2f}%, Both acc = {c_acc:.2f}%")

    return {
        "drone_counts": drone_counts,
        "sat_counts": sat_counts,
        "combined_counts": combined_counts,
        "total": total,
        "drone_paths": drone_paths,
        "drone_scenes": drone_scenes,
        "satellite_paths": satellite_paths,
        "satellite_scenes": satellite_scenes
    }


if __name__ == "__main__":
    #train()
    test()