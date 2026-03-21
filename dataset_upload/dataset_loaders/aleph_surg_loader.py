from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from dataset_upload.helpers import generate_unique_id


def _require_lerobot_dataset_class():
    """Import LeRobot lazily so unrelated dataset conversions still work."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
    except Exception as e:  # pragma: no cover - depends on local environment
        raise ImportError(
            "Aleph Surg conversion requires Hugging Face LeRobot. Install it in your env, e.g.\n"
            "  pip install lerobot\n"
            "Then re-run dataset conversion."
        ) from e
    return LeRobotDataset


def _require_decode_video_frames():
    try:
        from lerobot.datasets.video_utils import decode_video_frames  # type: ignore
    except Exception as e:  # pragma: no cover - depends on local environment
        raise ImportError(
            "Aleph Surg conversion requires Hugging Face LeRobot video utilities. Install lerobot and re-run."
        ) from e
    return decode_video_frames


def _discover_session_dirs(dataset_path: str | Path) -> list[Path]:
    """Discover Aleph Surg session directories from a session, dataset, or aggregate root."""
    root = Path(dataset_path).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    if (root / "meta" / "info.json").exists():
        return [root]

    direct_sessions = sorted(path for path in root.iterdir() if (path / "meta" / "info.json").exists())
    if direct_sessions:
        return direct_sessions

    nested_sessions: list[Path] = []
    for dataset_root in sorted(path for path in root.iterdir() if path.is_dir()):
        nested_sessions.extend(sorted(path for path in dataset_root.iterdir() if (path / "meta" / "info.json").exists()))

    if nested_sessions:
        return nested_sessions

    raise FileNotFoundError(f"Could not find meta/info.json under {root}")


def _session_dataset_id(session_dir: Path) -> str:
    return f"{session_dir.parent.name}/{session_dir.name}"


def _session_has_camera_videos(session_dir: Path, camera_key: str) -> bool:
    camera_dir = session_dir / "videos" / camera_key
    return camera_dir.exists() and any(camera_dir.rglob("*.mp4"))


def _filter_session_dirs_for_camera(
    session_dirs: list[Path], camera_key: str, session_allowlist: list[str] | None = None
) -> tuple[list[Path], dict[str, list[str]]]:
    allowlist = set(session_allowlist or [])
    session_id_to_path = {_session_dataset_id(path): path for path in session_dirs}

    missing_from_root = sorted(allowlist - set(session_id_to_path))
    if missing_from_root:
        raise ValueError(
            "Aleph Surg session_allowlist entries were not found under the dataset root: "
            + ", ".join(missing_from_root)
        )

    selected_session_dirs: list[Path] = []
    skipped = {"allowlist": [], "missing_camera": []}
    for session_dir in session_dirs:
        session_id = _session_dataset_id(session_dir)
        if allowlist and session_id not in allowlist:
            skipped["allowlist"].append(session_id)
            continue
        if not _session_has_camera_videos(session_dir, camera_key):
            skipped["missing_camera"].append(session_id)
            continue
        selected_session_dirs.append(session_dir)

    return selected_session_dirs, skipped


def _select_session_split(session_dirs: list[Path], split_name: str, eval_ratio: float, split_seed: int) -> list[Path]:
    """Create a deterministic session-level split shared across all cameras."""
    if split_name == "all":
        return session_dirs
    if not 0.0 < eval_ratio < 1.0:
        raise ValueError(f"eval_ratio must be between 0 and 1, got {eval_ratio}")
    if len(session_dirs) < 2:
        raise ValueError("Aleph Surg full-dataset splitting requires at least 2 sessions")

    ranked_sessions = sorted(
        session_dirs,
        key=lambda path: hashlib.sha1(f"{split_seed}:{_session_dataset_id(path)}".encode("utf-8")).hexdigest(),
    )
    eval_count = max(1, int(round(len(ranked_sessions) * eval_ratio)))
    eval_session_ids = {_session_dataset_id(path) for path in ranked_sessions[:eval_count]}

    if split_name == "eval":
        return [path for path in session_dirs if _session_dataset_id(path) in eval_session_ids]
    if split_name == "train":
        return [path for path in session_dirs if _session_dataset_id(path) not in eval_session_ids]

    raise ValueError(f"Unsupported split_name: {split_name}")


@lru_cache(maxsize=8)
def _load_lerobot_dataset_cached(dataset_root: str):
    """Load and memoize the local LeRobot dataset for repeated episode extraction."""
    LeRobotDataset = _require_lerobot_dataset_class()
    dataset_path = Path(dataset_root)
    dataset_id = f"{dataset_path.parent.name}/{dataset_path.name}"
    return LeRobotDataset(
        repo_id=dataset_id,
        root=dataset_path,
        revision="main",
        video_backend="pyav",
    )


class AlephSurgFrameLoader:
    """Lazy loader that extracts all frames for a single episode and camera."""

    def __init__(self, dataset_root: str, episode_index: int, camera_key: str) -> None:
        self.dataset_root = dataset_root
        self.episode_index = int(episode_index)
        self.camera_key = camera_key

    def __call__(self) -> np.ndarray:
        dataset = _load_lerobot_dataset_cached(self.dataset_root)
        decode_video_frames = _require_decode_video_frames()
        dataset._ensure_hf_dataset_loaded()
        episode_meta = dataset.meta.episodes[self.episode_index]
        start_idx = int(episode_meta["dataset_from_index"])
        end_idx = int(episode_meta["dataset_to_index"])
        from_timestamp = float(episode_meta[f"videos/{self.camera_key}/from_timestamp"])
        video_path = dataset.root / dataset.meta.get_video_file_path(self.episode_index, self.camera_key)
        if not video_path.exists():
            raise FileNotFoundError(
                f"Missing video for camera {self.camera_key} in episode {self.episode_index}: {video_path}"
            )

        shifted_timestamps: list[float] = []
        for frame_idx in range(start_idx, end_idx):
            frame_record = dataset.hf_dataset[frame_idx]
            current_ts = frame_record["timestamp"]
            current_ts = float(current_ts.item() if hasattr(current_ts, "item") else current_ts)
            shifted_timestamps.append(from_timestamp + current_ts)

        decoded_frames = decode_video_frames(video_path, shifted_timestamps, dataset.tolerance_s, dataset.video_backend)

        frames: list[np.ndarray] = []
        for frame in decoded_frames:
            frame_np = frame.numpy() if hasattr(frame, "numpy") else np.asarray(frame)
            if frame_np.dtype != np.uint8:
                frame_np = (frame_np * 255).clip(0, 255).astype(np.uint8)
            if frame_np.ndim == 3 and frame_np.shape[0] in (1, 3):
                frame_np = np.transpose(frame_np, (1, 2, 0))
            frames.append(frame_np)

        if not frames:
            raise ValueError(
                f"No frames found for episode {self.episode_index} with camera {self.camera_key} in {self.dataset_root}"
            )

        return np.stack(frames, axis=0)


def load_aleph_surg_dataset(
    dataset_path: str,
    camera_key: str,
    task_description: str,
    split_name: str = "train",
    eval_ratio: float = 0.15,
    split_seed: int = 42,
    episode_indices: list[int] | None = None,
    data_source: str = "aleph_surg",
    session_allowlist: list[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Load Aleph Surg trajectories grouped under a single full-task instruction."""
    session_dirs = _discover_session_dirs(dataset_path)
    camera_session_dirs, skipped_sessions = _filter_session_dirs_for_camera(
        session_dirs, camera_key=camera_key, session_allowlist=session_allowlist
    )
    if not camera_session_dirs:
        raise ValueError(f"No Aleph Surg sessions remain for camera {camera_key} under {dataset_path}")

    selected_session_dirs = _select_session_split(
        camera_session_dirs, split_name=split_name, eval_ratio=eval_ratio, split_seed=split_seed
    )

    task_data: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total_episodes = 0
    for session_dir in selected_session_dirs:
        lerobot_ds = _load_lerobot_dataset_cached(str(session_dir))

        available_camera_keys = list(getattr(lerobot_ds.meta, "camera_keys", []))
        if camera_key not in available_camera_keys:
            raise ValueError(f"Camera key {camera_key} not found in {session_dir}. Available cameras: {available_camera_keys}")

        available_episode_indices = list(range(len(lerobot_ds.meta.episodes)))
        selected_episode_indices = available_episode_indices if episode_indices is None else [int(i) for i in episode_indices]

        invalid_episode_indices = [idx for idx in selected_episode_indices if idx not in available_episode_indices]
        if invalid_episode_indices:
            raise ValueError(
                f"Episode indices {invalid_episode_indices} are out of range for {session_dir}; "
                f"available episodes are {available_episode_indices}"
            )

        for episode_index in selected_episode_indices:
            trajectory = {
                "id": generate_unique_id(),
                "task": task_description,
                "frames": AlephSurgFrameLoader(str(session_dir), episode_index, camera_key),
                "is_robot": True,
                "quality_label": "successful",
                "partial_success": 1.0,
                "data_source": data_source,
            }
            task_data[task_description].append(trajectory)
            total_episodes += 1

    print(
        f"Loaded {total_episodes} Aleph Surg episodes from {len(selected_session_dirs)} session(s) "
        f"for split={split_name} using camera {camera_key} as data_source={data_source}"
    )
    if skipped_sessions["allowlist"]:
        print(f"Skipped {len(skipped_sessions['allowlist'])} session(s) not present in session_allowlist")
    if skipped_sessions["missing_camera"]:
        print(f"Skipped {len(skipped_sessions['missing_camera'])} session(s) without video files for {camera_key}")
    return task_data
