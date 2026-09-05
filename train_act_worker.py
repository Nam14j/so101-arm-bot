#!/usr/bin/env python3
"""
train_act_worker.py — High-Performance PyTorch ACT Policy Trainer for SO-101 Arm
Loads dataset episodes, trains an Action Chunking Transformer, and saves deployable checkpoints.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import io

class JointActionDataset(Dataset):
    """Loads all .npz episode files from the dataset directory."""
    def __init__(self, dataset_dir, chunk_size=30, img_size=(224, 224)):
        self.dataset_dir = dataset_dir
        self.chunk_size = chunk_size
        self.img_size = img_size
        self.episodes = []
        self.samples = []

        self.transform = T.Compose([
            T.Resize(img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        ep_files = sorted([f for f in os.listdir(dataset_dir) if f.startswith("episode_") and f.endswith(".npz")])
        if not ep_files:
            raise ValueError(f"No episode files found in {dataset_dir}")

        print(f"[Dataset] Loading {len(ep_files)} episodes into RAM...", file=sys.stderr)
        for ep_file in ep_files:
            path = os.path.join(dataset_dir, ep_file)
            data = np.load(path, allow_pickle=True)
            qpos = data["qpos"]       # (T, 6)
            actions = data["actions"] # (T, 6)
            raw_images = data["images"]   # (T,) raw JPEG bytes or arrays
            
            # Trim stationary lead-in frames so the policy learns immediate dynamic reach
            deltas = np.linalg.norm(qpos - qpos[0], axis=1)
            move_steps = np.where(deltas > 2.5)[0]
            start_trim = max(0, (move_steps[0] - 2) if len(move_steps) > 0 else 0)

            qpos = qpos[start_trim:]
            actions = actions[start_trim:]
            raw_images = raw_images[start_trim:]

            T_len = len(qpos)
            if T_len < 10:
                continue

            # Pre-decode and transform images into RAM for 5x training acceleration
            cached_imgs = []
            for raw_img in raw_images:
                if isinstance(raw_img, bytes):
                    img = Image.open(io.BytesIO(raw_img)).convert("RGB")
                elif isinstance(raw_img, np.ndarray):
                    if raw_img.dtype == np.uint8:
                        img = Image.fromarray(raw_img).convert("RGB")
                    else:
                        img = Image.fromarray((raw_img * 255).astype(np.uint8)).convert("RGB")
                else:
                    img = Image.new("RGB", self.img_size, (0, 0, 0))
                cached_imgs.append(self.transform(img))

            ep_idx = len(self.episodes)
            self.episodes.append({
                "qpos": qpos,
                "actions": actions,
                "images": torch.stack(cached_imgs), # (T, 3, 224, 224)
                "length": T_len
            })

            # Create sample index tuples (ep_idx, t)
            for t in range(T_len):
                self.samples.append((ep_idx, t))

        # Compute dataset statistics for proper z-score normalization (ACT standard)
        all_qpos = np.concatenate([ep["qpos"] for ep in self.episodes], axis=0)
        all_actions = np.concatenate([ep["actions"] for ep in self.episodes], axis=0)
        self.qpos_mean = np.mean(all_qpos, axis=0).astype(np.float32)
        self.qpos_std = np.clip(np.std(all_qpos, axis=0).astype(np.float32), 1e-2, None)
        self.action_mean = np.mean(all_actions, axis=0).astype(np.float32)
        self.action_std = np.clip(np.std(all_actions, axis=0).astype(np.float32), 1e-2, None)

        print(f"[Dataset] Pre-cached {len(self.episodes)} episodes ({len(self.samples)} steps) in high-speed RAM. Ready for turbo training!", file=sys.stderr)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ep_idx, t = self.samples[idx]
        ep = self.episodes[ep_idx]
        T_len = ep["length"]

        # Current normalized state
        raw_qpos = ep["qpos"][t]
        norm_qpos = (raw_qpos - self.qpos_mean) / self.qpos_std
        current_qpos = torch.tensor(norm_qpos, dtype=torch.float32)

        # Pre-cached transformed image tensor from RAM (instant 0ms retrieval)
        img_tensor = ep["images"][t]

        # Future action chunk of length self.chunk_size
        end_idx = min(t + self.chunk_size, T_len)
        action_slice = ep["actions"][t:end_idx]
        
        # Pad with last action if near end of episode
        if len(action_slice) < self.chunk_size:
            pad_len = self.chunk_size - len(action_slice)
            last_act = action_slice[-1] if len(action_slice) > 0 else ep["actions"][-1]
            padding = np.repeat(last_act[np.newaxis, :], pad_len, axis=0)
            action_chunk = np.concatenate([action_slice, padding], axis=0)
        else:
            action_chunk = action_slice

        # Normalize action targets
        norm_action_chunk = (action_chunk - self.action_mean) / self.action_std
        action_tensor = torch.tensor(norm_action_chunk, dtype=torch.float32) # (chunk_size, 6)

        return {
            "image": img_tensor,
            "qpos": current_qpos,
            "action": action_tensor
        }


class SimpleACTPolicy(nn.Module):
    """
    Action Chunking Transformer (ACT) for Visual Imitation Learning.
    Encodes camera image via ResNet-18 backbone + state embedding,
    and decodes future action chunks with Transformer decoder.
    """
    def __init__(self, action_dim=6, state_dim=6, chunk_size=30, d_model=256, nhead=4, num_layers=3):
        super().__init__()
        import torchvision.models as models
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.d_model = d_model

        # Visual Backbone: ResNet-18
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]) # Output (B, 512, 1, 1)
        self.img_proj = nn.Linear(512, d_model)

        # State projection
        self.qpos_proj = nn.Linear(state_dim, d_model)

        # Action query tokens
        self.query_embed = nn.Embedding(chunk_size, d_model)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=512, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Action head
        self.action_head = nn.Linear(d_model, action_dim)

    def forward(self, images, qpos):
        # images: (B, 3, 224, 224), qpos: (B, 6)
        B = images.shape[0]

        # Visual features
        feat = self.backbone(images).flatten(1) # (B, 512)
        feat_emb = self.img_proj(feat).unsqueeze(1) # (B, 1, d_model)

        # Qpos features
        qpos_emb = self.qpos_proj(qpos).unsqueeze(1) # (B, 1, d_model)

        # Memory = [Image Token, State Token]
        memory = torch.cat([feat_emb, qpos_emb], dim=1) # (B, 2, d_model)

        # Queries = (B, chunk_size, d_model)
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)

        # Decode actions
        out = self.decoder(tgt=queries, memory=memory) # (B, chunk_size, d_model)
        actions = self.action_head(out) # (B, chunk_size, action_dim)
        return actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--steps", type=int, default=25000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--chunk_size", type=int, default=30)
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint .pt to resume from")
    args = parser.parse_args()

    # Device selection
    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"[Train] Initializing dataset from {args.dataset_dir} on {device}...", file=sys.stderr)
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        dataset = JointActionDataset(args.dataset_dir, chunk_size=args.chunk_size)
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}), flush=True)
        sys.exit(1)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    model = SimpleACTPolicy(action_dim=6, state_dim=6, chunk_size=args.chunk_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=1e-5)

    start_step = 1
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"[Train] Resuming from checkpoint: {args.resume_from}", file=sys.stderr)
        try:
            ckpt = torch.load(args.resume_from, map_location=device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state_dict)
            if "optimizer_state_dict" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                except Exception as oe:
                    print(f"[Train] Warning: could not restore optimizer: {oe}", file=sys.stderr)
            saved_step = ckpt.get("step", 0)
            start_step = saved_step + 1
            # Advance scheduler
            for _ in range(saved_step):
                scheduler.step()
            print(f"[Train] Restored model at step {saved_step}. Continuing to {args.steps} steps...", file=sys.stderr)
        except Exception as ex:
            print(f"[Train] Error loading resume checkpoint: {ex}", file=sys.stderr)

    print(f"[Train] Model ready. Starting training from step {start_step} to {args.steps}...", file=sys.stderr)
    print(json.dumps({"status": "started", "start_step": start_step, "total_steps": args.steps, "device": str(device)}), flush=True)

    data_iter = iter(dataloader)
    start_time = time.time()
    running_loss = 0.0

    for step in range(start_step, args.steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        images = batch["image"].to(device)
        qpos = batch["qpos"].to(device)
        target_actions = batch["action"].to(device)

        optimizer.zero_grad()
        pred_actions = model(images, qpos)
        loss = F.l1_loss(pred_actions, target_actions) + 0.5 * F.mse_loss(pred_actions, target_actions)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        # Log progress every 20 steps
        if step % 20 == 0 or step == start_step:
            avg_loss = running_loss / (20 if (step - start_step) > 0 else 1)
            running_loss = 0.0
            elapsed = time.time() - start_time
            steps_done = step - start_step + 1
            steps_per_sec = steps_done / max(1e-3, elapsed)
            remaining_sec = (args.steps - step) / max(1e-3, steps_per_sec)
            eta_str = time.strftime("%H:%M:%S", time.gmtime(remaining_sec))

            log_entry = {
                "status": "progress",
                "step": step,
                "total_steps": args.steps,
                "loss": round(avg_loss, 4),
                "lr": float(f"{scheduler.get_last_lr()[0]:.2e}"),
                "fps": round(steps_per_sec * args.batch_size, 1),
                "eta": eta_str
            }
            print(json.dumps(log_entry), flush=True)

        # Save Checkpoints
        if step % 5000 == 0 or step == args.steps:
            stats_payload = {
                "qpos_mean": dataset.qpos_mean.tolist(),
                "qpos_std": dataset.qpos_std.tolist(),
                "action_mean": dataset.action_mean.tolist(),
                "action_std": dataset.action_std.tolist()
            }
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_{step}.pt")
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "stats": stats_payload,
                "args": vars(args)
            }, ckpt_path)
            
            # Save latest symlink/copy
            latest_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "stats": stats_payload,
                "args": vars(args)
            }, latest_path)

            print(json.dumps({"status": "saved_checkpoint", "path": ckpt_path, "step": step}), flush=True)

    print(json.dumps({"status": "completed", "total_steps": args.steps, "output_dir": args.output_dir}), flush=True)
    print(f"[Train] Finished successfully!", file=sys.stderr)

if __name__ == "__main__":
    main()
