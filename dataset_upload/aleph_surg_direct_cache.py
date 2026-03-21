#!/usr/bin/env python3
"""
Direct-to-cache converter for Aleph Surg datasets.
Bypasses generate_hf_dataset + preprocess_datasets by reading LeRobot MP4s
with PyAV and writing the RoboMeter preprocessing cache directly.
"""

from __future__ import annotations

import datetime
import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import av
import numpy as np
from datasets import Dataset, Features, Sequence, Value
from pyrallis import wrap
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from dataset_upload.dataset_loaders.aleph_surg_loader import (
    _discover_session_dirs,
    _filter_session_dirs_for_camera,
    _select_session_split,
    _session_dataset_id,
)


@dataclass
class DirectCacheConfig:
    dataset_path: str = ""
    camera_key: str = ""
    task_description: str = ""
    split_name: str = "train"
    eval_ratio: float = 0.15
    split_seed: int = 42
    session_allowlist: list[str] = field(default_factory=list)
    data_source: str = ""
    max_frames: int = 100
    cache_dir: str = ""
    cache_key: str = ""


def _load_session_video_paths(
    session_dir: Path, camera_key: str
) -> list[Path]:
    """Load LeRobot metadata for a session, return video paths per episode, then free memory."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset_id = _session_dataset_id(session_dir)
    ds = LeRobotDataset(
        repo_id=dataset_id,
        root=session_dir,
        revision="main",
        video_backend="pyav",
    )
    num_episodes = len(ds.meta.episodes)
    video_paths = []
    for ep_idx in range(num_episodes):
        vp = ds.root / ds.meta.get_video_file_path(ep_idx, camera_key)
        video_paths.append(vp)
    del ds
    return video_paths


def _sample_frames(video_path: Path, max_frames: int) -> np.ndarray:
    """Sample up to *max_frames* uniformly from a video file using PyAV."""
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    total = stream.frames
    if total <= 0:
        all_frames = [f.to_ndarray(format="rgb24") for f in container.decode(stream)]
        container.close()
        total = len(all_frames)
        if total <= max_frames:
            return np.stack(all_frames)
        indices = {int(i * total / max_frames) for i in range(max_frames)}
        selected = [f for i, f in enumerate(all_frames) if i in indices]
        return np.stack(selected)

    if total <= max_frames:
        indices = set(range(total))
    else:
        indices = {int(i * total / max_frames) for i in range(max_frames)}

    frames = []
    for i, frame in enumerate(container.decode(stream)):
        if i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))
        if len(frames) == len(indices):
            break
    container.close()
    return np.stack(frames)


def _build_index_mappings(
    n: int,
    task_description: str,
    data_source: str,
) -> dict:
    all_indices = list(range(n))
    return {
        "robot_trajectories": all_indices,
        "human_trajectories": [],
        "optimal_by_task": {task_description: all_indices},
        "suboptimal_by_task": {},
        "quality_indices": {"successful": all_indices},
        "task_indices": {task_description: all_indices},
        "source_indices": {data_source: all_indices},
        "partial_success_indices": {},
    }


@wrap()
def main(cfg: DirectCacheConfig):
    dataset_path = cfg.dataset_path
    camera_key = cfg.camera_key
    task_description = cfg.task_description
    split_name = cfg.split_name
    eval_ratio = cfg.eval_ratio
    split_seed = cfg.split_seed
    session_allowlist = cfg.session_allowlist or None
    data_source = cfg.data_source
    max_frames = cfg.max_frames
    cache_dir = cfg.cache_dir
    cache_key = cfg.cache_key

    if not dataset_path:
        raise ValueError("dataset_path is required")
    if not camera_key:
        raise ValueError("camera_key is required")
    if not task_description:
        raise ValueError("task_description is required")
    if not cache_dir:
        raise ValueError("cache_dir is required")
    if not cache_key:
        raise ValueError("cache_key is required")

    # --- discover, filter, split sessions ---
    print(f"Discovering sessions under {dataset_path} ...")
    session_dirs = _discover_session_dirs(dataset_path)
    print(f"  Found {len(session_dirs)} session(s) total")

    camera_session_dirs, skipped = _filter_session_dirs_for_camera(
        session_dirs, camera_key=camera_key, session_allowlist=session_allowlist
    )
    if skipped["allowlist"]:
        print(f"  Skipped {len(skipped['allowlist'])} session(s) not in allowlist")
    if skipped["missing_camera"]:
        print(
            f"  Skipped {len(skipped['missing_camera'])} session(s) without video for {camera_key}"
        )
    if not camera_session_dirs:
        raise ValueError(
            f"No sessions remain for camera {camera_key} under {dataset_path}"
        )

    selected_session_dirs = _select_session_split(
        camera_session_dirs,
        split_name=split_name,
        eval_ratio=eval_ratio,
        split_seed=split_seed,
    )
    print(
        f"  {len(selected_session_dirs)} session(s) selected for split={split_name}"
    )

    # --- language embedding (compute once) ---
    print("Computing language embedding ...")
    lang_model = SentenceTransformer("all-MiniLM-L6-v2")
    lang_vector = lang_model.encode(task_description)
    del lang_model

    # --- create output directories ---
    cache_root = os.path.join(cache_dir, cache_key)
    frames_dir = os.path.join(cache_root, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(os.path.join(cache_root, "processed_dataset"), exist_ok=True)

    # --- process sessions & episodes ---
    metadata_rows: list[dict] = []
    total_skipped = 0

    for session_dir in tqdm(
        selected_session_dirs, desc="Sessions", unit="session"
    ):
        session_id = _session_dataset_id(session_dir)
        print(f"\nLoading metadata for {session_id} ...")
        try:
            video_paths = _load_session_video_paths(session_dir, camera_key)
        except Exception as exc:
            print(f"  ERROR loading session {session_id}: {exc}  — skipping")
            continue

        for ep_idx, video_path in enumerate(
            tqdm(
                video_paths,
                desc=f"  Episodes ({session_id})",
                unit="ep",
                leave=False,
            )
        ):
            if not video_path.exists():
                print(
                    f"  WARNING: missing video {video_path} — skipping episode {ep_idx}"
                )
                total_skipped += 1
                continue

            try:
                frames = _sample_frames(video_path, max_frames)
            except Exception as exc:
                print(
                    f"  ERROR sampling frames from {video_path}: {exc} — skipping"
                )
                total_skipped += 1
                continue

            frames_shape = frames.shape  # (T, H, W, C)

            traj_id = str(uuid.uuid4())
            npz_filename = f"trajectory_{traj_id}.npz"
            npz_path = os.path.join(frames_dir, npz_filename)
            np.savez_compressed(
                npz_path,
                frames=frames,
                shape=frames_shape,
                num_frames=frames_shape[0],
            )
            del frames

            metadata_rows.append(
                {
                    "id": traj_id,
                    "task": task_description,
                    "lang_vector": lang_vector.tolist(),
                    "data_source": data_source,
                    "frames": npz_path,
                    "frames_shape": list(frames_shape),
                    "num_frames": int(frames_shape[0]),
                    "frames_processed": True,
                    "is_robot": True,
                    "quality_label": "successful",
                    "partial_success": 1.0,
                }
            )

    if not metadata_rows:
        raise RuntimeError(
            "No trajectories were produced. Check dataset path and camera key."
        )

    n = len(metadata_rows)
    print(f"\nProcessed {n} trajectories ({total_skipped} skipped)")

    # --- index mappings ---
    index_mappings = _build_index_mappings(n, task_description, data_source)
    index_mappings_path = os.path.join(cache_root, "index_mappings.json")
    with open(index_mappings_path, "w") as f:
        json.dump(index_mappings, f, indent=2)
    print(f"Saved index_mappings.json -> {index_mappings_path}")

    # --- HuggingFace Dataset ---
    features = Features(
        {
            "id": Value("string"),
            "task": Value("string"),
            "lang_vector": Sequence(Value("float32")),
            "data_source": Value("string"),
            "frames": Value("string"),
            "frames_shape": Sequence(Value("int64")),
            "num_frames": Value("int64"),
            "frames_processed": Value("bool"),
            "is_robot": Value("bool"),
            "quality_label": Value("string"),
            "partial_success": Value("float32"),
        }
    )
    data_dict = {key: [row[key] for row in metadata_rows] for key in metadata_rows[0]}
    dataset = Dataset.from_dict(data_dict, features=features)
    dataset_path_out = os.path.join(cache_root, "processed_dataset")
    dataset.save_to_disk(dataset_path_out)
    print(f"Saved HF Dataset ({len(dataset)} rows) -> {dataset_path_out}")

    # --- dataset_info.json ---
    dataset_info = {
        "dataset_path": dataset_path,
        "subset": split_name,
        "total_trajectories": n,
        "cache_timestamp": str(datetime.datetime.now()),
        "config_hash": "direct_cache",
    }
    info_path = os.path.join(cache_root, "dataset_info.json")
    with open(info_path, "w") as f:
        json.dump(dataset_info, f, indent=2)
    print(f"Saved dataset_info.json -> {info_path}")

    # --- summary ---
    print("\n=== Direct cache complete ===")
    print(f"  Cache root : {cache_root}")
    print(f"  Trajectories: {n}")
    print(f"  Skipped     : {total_skipped}")
    print(f"  Split       : {split_name}")
    print(f"  Camera      : {camera_key}")
    print(f"  Data source : {data_source}")


if __name__ == "__main__":
    main()
