"""
基于显式/隐式双重记忆的长视频时空理解框架
用于处理室内场景的第一人称视频，支持QA、Navigation等任务
"""

import os
import argparse
from typing import List, Optional
from PIL import Image
import cv2
import torch

from model.qwen_based import Qwen3VLWrapper, Qwen2_5_VLWrapper, ReasoningModule
from model.memory_module import ExplicitMemoryManager, ImplicitMemoryBank
from model.utils import process_streaming_video


def load_video_frames(video_path: str, max_frames: Optional[int] = None, fps: int = 30) -> List[Image.Image]:
    """
    从视频文件加载帧
    
    Args:
        video_path: 视频文件路径
        max_frames: 最大帧数，None表示加载所有帧
        fps: 目标帧率，用于采样
    
    Returns:
        frames: PIL Image对象列表
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = max(1, int(original_fps / fps))  # 根据目标fps采样
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 按目标fps采样
        if frame_count % frame_skip == 0:
            # BGR转RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 转换为PIL Image
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
            
            if max_frames and len(frames) >= max_frames:
                break
        
        frame_count += 1
    
    cap.release()
    print(f"Loaded {len(frames)} frames from {video_path}")
    return frames


def main():
    parser = argparse.ArgumentParser(description="Dual Memory Video Understanding Framework")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video file")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL-2B-Instruct", 
                       help="HuggingFace model name or path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run the model on")
    parser.add_argument("--window_size", type=int, default=16, help="Sliding window size (frames)")
    parser.add_argument("--window_stride", type=int, default=8, help="Sliding window stride (frames)")
    parser.add_argument("--update_explicit_every", type=int, default=3, 
                       help="Update explicit memory every N windows")
    parser.add_argument("--recluster_every", type=int, default=10,
                       help="Recluster implicit memory every N windows")
    parser.add_argument("--max_frames", type=int, default=None,
                       help="Maximum number of frames to process (for testing)")
    parser.add_argument("--use_qwen3", action="store_true",
                       help="Use Qwen3VLWrapper instead of Qwen2_5_VLWrapper")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Dual Memory Video Understanding Framework")
    print("=" * 60)
    
    # 1. 加载模型
    print("\n[1/4] Loading model...")
    if args.use_qwen3:
        qwen = Qwen3VLWrapper(model_name=args.model_name, device=args.device)
        print(f"Using Qwen3VLWrapper with model: {args.model_name}")
    else:
        qwen = Qwen2_5_VLWrapper(model_name=args.model_name, device=args.device)
        print(f"Using Qwen2_5_VLWrapper with model: {args.model_name}")
    
    # 2. 初始化记忆模块
    print("\n[2/4] Initializing memory modules...")
    explicit_mgr = ExplicitMemoryManager()
    implicit_bank = ImplicitMemoryBank(n_clusters=10)  # 可以根据需要调整聚类数
    reasoner = ReasoningModule(qwen, explicit_mgr, implicit_bank)
    print("Memory modules initialized.")
    
    # 3. 加载视频
    print(f"\n[3/4] Loading video from {args.video_path}...")
    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"Video file not found: {args.video_path}")
    
    frames = load_video_frames(args.video_path, max_frames=args.max_frames)
    if not frames:
        raise ValueError("No frames loaded from video")
    
    # 4. 流式处理视频
    print(f"\n[4/4] Processing streaming video...")
    print(f"  - Total frames: {len(frames)}")
    print(f"  - Window size: {args.window_size}")
    print(f"  - Window stride: {args.window_stride}")
    print(f"  - Update explicit memory every: {args.update_explicit_every} windows")
    print(f"  - Recluster every: {args.recluster_every} windows")
    
    process_streaming_video(
        frames_stream=frames,
        qwen=qwen,
        explicit_mgr=explicit_mgr,
        implicit_bank=implicit_bank,
        window_size=args.window_size,
        window_stride=args.window_stride,
        update_explicit_every=args.update_explicit_every,
        recluster_every=args.recluster_every
    )
    
    # 5. 显示显式记忆（空间认知图）
    print("\n" + "=" * 60)
    print("Explicit Memory (Spatial Relation Map):")
    print("=" * 60)
    print(explicit_mgr.to_json_str())
    
    # 6. 显示隐式记忆统计
    print("\n" + "=" * 60)
    print("Implicit Memory Statistics:")
    print("=" * 60)
    print(f"Total memory items: {implicit_bank.get_size()}")
    cluster_centers = implicit_bank.get_cluster_centers()
    print(f"Cluster centers shape: {cluster_centers.shape}")
    
    # 7. 示例问答
    print("\n" + "=" * 60)
    print("Question Answering Examples:")
    print("=" * 60)
    
    example_questions = [
        "Where is the washing machine?",
        "What rooms are connected to the kitchen?",
        "Describe the spatial layout of the environment.",
    ]
    
    for q in example_questions:
        print(f"\nQ: {q}")
        try:
            answer = reasoner.answer(q, top_k=5)
            print(f"A: {answer}")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "=" * 60)
    print("Processing completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
