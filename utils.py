import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
import numpy as np
import torch
from PIL import Image
from .qwen_based import Qwen3VLWrapper, Qwen2_5_VLWrapper
from .memory_module import ExplicitMemoryManager, ImplicitMemoryBank


def process_streaming_video(
    frames_stream: Union[List[Image.Image], Any],  # 可迭代的帧列表 or generator
    qwen: Union[Qwen3VLWrapper, Qwen2_5_VLWrapper],
    explicit_mgr: ExplicitMemoryManager,
    implicit_bank: ImplicitMemoryBank,
    window_size: int = 16,
    window_stride: int = 8,
    update_explicit_every: int = 3,
    recluster_every: int = 10,  # 每隔多少个窗口重新聚类
    frame_interval_sec: float = 0.033  # 假设30fps，每帧约0.033秒
):
    """
    处理流式视频，同时构建显式记忆和隐式记忆
    
    Args:
        frames_stream: 视频帧列表或生成器（PIL Image对象）
        qwen: Qwen模型包装器
        explicit_mgr: 显式记忆管理器
        implicit_bank: 隐式记忆库
        window_size: 滑动窗口大小（帧数）
        window_stride: 滑动窗口步长（帧数）
        update_explicit_every: 每隔多少个窗口更新一次显式记忆
        recluster_every: 每隔多少个窗口重新聚类一次隐式记忆
        frame_interval_sec: 每帧的时间间隔（秒）
    """
    buffer = []
    window_index = 0
    
    # 如果frames_stream是列表，转换为迭代器
    if isinstance(frames_stream, list):
        frames_iter = iter(frames_stream)
    else:
        frames_iter = frames_stream
    
    for t, frame in enumerate(frames_iter):
        # 确保frame是PIL Image
        if not isinstance(frame, Image.Image):
            try:
                frame = Image.fromarray(frame)
            except:
                raise ValueError(f"Frame at index {t} cannot be converted to PIL Image")
        
        buffer.append(frame)

        # 每 stride 长度处理一个滑窗
        if len(buffer) >= window_size:
            clip_frames = buffer[-window_size:]  # 当前窗口

            # === Step 1. 取全局视觉特征，存入 implicit memory ===
            enc = qwen.encode_clip(clip_frames)
            global_feat = enc["global_feature"]     # (D,)
            timestamp = t * frame_interval_sec  # 转换为秒
            implicit_bank.append(global_feat, index=window_index,
                                 timestamp=timestamp, meta={"window_index": window_index})
            window_index += 1

            # === Step 2. 定期更新 explicit memory ===
            if window_index % update_explicit_every == 0:
                print(f"Updating explicit memory at window {window_index}...")
                current_json = explicit_mgr.to_json_str()
                new_json = qwen.update_explicit_memory(clip_frames, current_json)
                explicit_mgr.load_from_json_str(new_json)

            # === Step 3. 定期重新聚类 implicit memory ===
            if window_index % recluster_every == 0 and implicit_bank.get_size() > 0:
                print(f"Reclustering implicit memory at window {window_index}...")
                implicit_bank.cluster()

            # 滑动：移除前 stride 帧
            if window_stride > 0 and len(buffer) >= window_size + window_stride:
                buffer = buffer[window_stride:]
    
    # 处理完成后，进行最后一次聚类
    if implicit_bank.get_size() > 0:
        print("Performing final clustering...")
        implicit_bank.cluster()
    
    print(f"Streaming video processing completed. Total windows: {window_index}")
