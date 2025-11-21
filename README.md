# Dual Memory Video Understanding Framework

基于显式/隐式双重记忆的长视频时空理解框架，用于处理室内场景的第一人称视频，支持QA、Navigation、Captioning等任务。

## 功能特点

1. **显式记忆（Explicit Memory）**：维护一个空间认知图（JSON格式），包含房间连通关系和物体相对位置信息
2. **隐式记忆（Implicit Memory）**：存储视频片段的视觉特征，通过聚类提取全局特征
3. **流式处理**：使用滑动窗口处理长视频（10-15分钟）
4. **多任务支持**：支持问答（QA）、导航（Navigation）、描述生成（Captioning）等任务

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本使用

```bash
python main.py --video_path /path/to/your/video.mp4
```

### 完整参数示例

```bash
python main.py \
    --video_path /path/to/video.mp4 \
    --model_name Qwen/Qwen2-VL-2B-Instruct \
    --device cuda \
    --window_size 16 \
    --window_stride 8 \
    --update_explicit_every 3 \
    --recluster_every 10 \
    --max_frames 1000
```

### 参数说明

- `--video_path`: 输入视频文件路径（必需）
- `--model_name`: HuggingFace模型名称或路径（默认：Qwen/Qwen2-VL-2B-Instruct）
- `--device`: 运行设备，cuda或cpu（默认：自动检测）
- `--window_size`: 滑动窗口大小，单位：帧数（默认：16）
- `--window_stride`: 滑动窗口步长，单位：帧数（默认：8）
- `--update_explicit_every`: 每隔多少个窗口更新一次显式记忆（默认：3）
- `--recluster_every`: 每隔多少个窗口重新聚类一次隐式记忆（默认：10）
- `--max_frames`: 最大处理帧数，用于测试（默认：None，处理所有帧）
- `--use_qwen3`: 使用Qwen3VLWrapper而不是Qwen2_5_VLWrapper

## 项目结构

```
DualMemory/
├── main.py                 # 主程序入口
├── model/
│   ├── __init__.py        # 包初始化
│   ├── qwen_based.py     # Qwen模型包装器和推理模块
│   ├── memory_module.py  # 显式和隐式记忆管理
│   └── utils.py          # 流式视频处理工具
├── requirements.txt       # 依赖包列表
└── README.md             # 本文件
```

## 工作流程

1. **视频加载**：从视频文件加载帧序列
2. **流式处理**：
   - 使用滑动窗口处理视频帧
   - 提取每个窗口的视觉特征，存入隐式记忆
   - 定期更新显式记忆（空间认知图）
   - 定期对隐式记忆进行聚类
3. **推理**：
   - 使用问题文本从隐式记忆检索相关证据
   - 结合显式记忆和检索到的证据进行推理
   - 生成答案

## 显式记忆格式

显式记忆以JSON格式存储空间认知图：

```json
{
  "rooms": [
    {
      "name": "Kitchen",
      "objects": ["Fridge", "Sink 1"]
    },
    {
      "name": "Living Room",
      "objects": ["Sofa", "TV", "Desk 0"]
    }
  ],
  "objects": [],
  "connections": []
}
```

## 代码示例

### 使用Python API

```python
from model import Qwen2_5_VLWrapper, ExplicitMemoryManager, ImplicitMemoryBank, ReasoningModule, process_streaming_video
from PIL import Image

# 初始化模型和记忆模块
qwen = Qwen2_5_VLWrapper(model_name="Qwen/Qwen2-VL-2B-Instruct")
explicit_mgr = ExplicitMemoryManager()
implicit_bank = ImplicitMemoryBank(n_clusters=10)
reasoner = ReasoningModule(qwen, explicit_mgr, implicit_bank)

# 加载视频帧（示例）
frames = [...]  # List[Image.Image]

# 处理流式视频
process_streaming_video(
    frames_stream=frames,
    qwen=qwen,
    explicit_mgr=explicit_mgr,
    implicit_bank=implicit_bank,
    window_size=16,
    window_stride=8,
    update_explicit_every=3
)

# 进行问答
question = "Where is the washing machine?"
answer = reasoner.answer(question, top_k=5)
print(f"Q: {question}")
print(f"A: {answer}")
```

## 注意事项

1. **模型要求**：需要安装Qwen-VL模型，建议使用HuggingFace上的预训练模型
2. **内存要求**：处理长视频需要较大内存，建议使用GPU加速
3. **视频格式**：支持常见的视频格式（mp4, avi等），使用OpenCV加载
4. **性能优化**：
   - 可以通过`--max_frames`限制处理帧数进行测试
   - 调整`window_size`和`window_stride`平衡处理速度和精度
   - 调整`update_explicit_every`和`recluster_every`控制更新频率

## 依赖包

主要依赖：
- torch >= 2.0.0
- transformers >= 4.37.0
- qwen-vl-utils >= 0.0.1
- opencv-python >= 4.5.0
- scikit-learn >= 1.0.0

完整列表请参见 `requirements.txt`

## 许可证

本项目仅供学术研究使用。

## 作者

毕设项目 - Dual Memory Video Understanding Framework

