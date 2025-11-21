import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, TYPE_CHECKING
import numpy as np
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# 尝试导入 Qwen2.5-VL 模型类（如果可用）
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
    HAS_QWEN2_5_VL = True
except ImportError:
    # 如果 transformers 版本较旧，可能没有这个类
    HAS_QWEN2_5_VL = False
    Qwen2_5_VLForConditionalGeneration = None

# 延迟导入以避免循环依赖
if TYPE_CHECKING:
    from .memory_module import MemoryItem, ExplicitMemoryManager, ImplicitMemoryBank


def _get_model_class(model_name: str):
    """
    根据模型名称返回正确的模型类
    不使用 AutoModel，而是根据模型名称手动判断
    """
    model_name_lower = model_name.lower()
    
    # 检查是否是 Qwen2.5-VL
    if "qwen2.5" in model_name_lower or "qwen2_5" in model_name_lower:
        if HAS_QWEN2_5_VL and Qwen2_5_VLForConditionalGeneration is not None:
            return Qwen2_5_VLForConditionalGeneration
        else:
            raise ImportError(
                f"Qwen2.5-VL model detected but Qwen2_5_VLForConditionalGeneration is not available. "
                f"Please upgrade transformers: pip install --upgrade transformers"
            )
    else:
        # 默认使用 Qwen2-VL
        return Qwen2VLForConditionalGeneration


class Qwen3VLWrapper:
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-2B-Instruct", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        初始化Qwen3-VL模型包装器
        Args:
            model_name: HuggingFace模型名称或路径
            device: 设备类型
        """
        self.device = device
        self.model_name = model_name
        
        # 根据模型名称选择正确的模型类
        model_class = _get_model_class(model_name)
        self.model = model_class.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map=device,
            trust_remote_code=True
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        
    @torch.no_grad()
    def encode_clip(self, frames: List[Image.Image]) -> Dict[str, torch.Tensor]:
        """
        输入: 一个滑动窗口内的若干帧 (list of PIL images)
        输出:
            - patch_tokens: (N_tokens, D)
            - global_feature: (D,)  # pooled token，可当作 video-CLS
        """
        # 处理多帧图像
        if not frames:
            raise ValueError("frames list cannot be empty")
        
        # 对于 Qwen2.5-VL，processor 需要文本输入
        # 检查是否是 Qwen2.5-VL processor（通过检查 processor 类型）
        is_qwen2_5 = "qwen2_5" in str(type(self.processor)).lower() or "qwen2.5" in str(type(self.processor)).lower()
        
        if is_qwen2_5:
            # Qwen2.5-VL 需要文本输入，使用一个简单的占位符文本
            # 使用列表格式，每个图像对应一个文本
            text_list = [""] * len(frames)
            inputs = self.processor(
                text=text_list,
                images=frames,
                return_tensors="pt",
                padding=True
            ).to(self.device)
        else:
            # Qwen2-VL 可以直接处理图像
            try:
                inputs = self.processor(
                    images=frames,
                    return_tensors="pt"
                ).to(self.device)
            except Exception as e:
                # 如果失败，尝试提供文本输入
                text_list = [""] * len(frames)
                inputs = self.processor(
                    text=text_list,
                    images=frames,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
        
        # 获取vision encoder输出
        # 兼容 Qwen2-VL 和 Qwen2.5-VL 的不同结构
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'visual'):
            vision_outputs = self.model.model.visual(**inputs['pixel_values'])
        elif hasattr(self.model, 'visual'):
            vision_outputs = self.model.visual(**inputs['pixel_values'])
        else:
            raise AttributeError(f"Cannot find vision encoder in model {self.model_name}")
        
        # 提取patch tokens和全局特征
        # vision_outputs.last_hidden_state: (batch, num_patches, hidden_size)
        patch_tokens = vision_outputs.last_hidden_state  # (B, N_patches, D)
        
        # 全局特征：对patch tokens进行平均池化
        global_feature = patch_tokens.mean(dim=1).squeeze(0)  # (D,)
        
        # 展平patch tokens用于后续处理
        patch_tokens_flat = patch_tokens.view(-1, patch_tokens.size(-1))  # (B*N_patches, D)
        
        return {
            "patch_tokens": patch_tokens_flat,
            "global_feature": global_feature
        }

    @torch.no_grad()
    def update_explicit_memory(self, frames: List[Image.Image], current_memory_json: str) -> str:
        """
        使用 prompt + 当前 explicit map + 新视频片段，让 Qwen 输出新的 JSON。
        返回新的 explicit memory (string 格式，保持 JSON schema)
        """
        prompt = (
            "You are an indoor spatial map maintainer.\n"
            "Here is the current spatial relation map in JSON:\n"
            f"{current_memory_json}\n\n"
            "Now you observe a new egocentric video clip. "
            "Please update or correct the map according to this clip, "
            "and output ONLY the updated JSON with the same schema."
        )
        
        # 构建多模态输入
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": frame} for frame in frames
                ] + [{"type": "text", "text": prompt}]
            }
        ]
        
        # 处理vision信息
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # 生成回复
        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        # 提取JSON
        new_json_str = self._extract_json_from_text(output_text)
        return new_json_str

    def _extract_json_from_text(self, text: str) -> str:
        # 简单的提取逻辑：找到第一个 { 和 最后一个 }
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            return text[start:end]
        except ValueError:
            # 如果失败，就保留原来的（保守策略）
            return text

    @torch.no_grad()
    def get_text_embedding(self, question: str) -> torch.Tensor:
        """
        用 Qwen3-VL 的 text encoder / decoder hidden state 取一个问题的向量表示。
        training-free 检索备用。
        """
        # 使用processor处理文本
        inputs = self.processor(
            text=question,
            return_tensors="pt"
        ).to(self.device)
        
        # 获取文本embedding
        # 使用模型的embedding层
        text_embeddings = self.model.model.embed_tokens(inputs.input_ids)  # (B, L, D)
        
        # 对序列维度求平均作为文本表示
        emb = text_embeddings.mean(dim=1).squeeze(0)  # (D,)
        return emb

    @torch.no_grad()
    def answer_with_evidence(
        self,
        question: str,
        explicit_memory_json: str,
        evidence_global_features: List[torch.Tensor]
    ) -> str:
        """
        使用：问题 + 显式记忆(JSON) + 若干检索到的 implicit memory tokens 进行最终推理。
        注意：由于Qwen-VL的API限制，这里我们将evidence features转换为伪图像表示
        实际应用中可能需要更复杂的融合策略
        """
        prompt = (
            "You are an assistant that answers questions about an indoor environment.\n"
            "Here is the current spatial map in JSON:\n"
            f"{explicit_memory_json}\n\n"
            f"Question: {question}\n"
            "Please answer concisely."
        )
        
        # 构建消息
        content = [{"type": "text", "text": prompt}]
        
        # 如果有evidence features，可以尝试将其转换为图像表示
        # 这里简化处理：将features信息编码到prompt中
        if evidence_global_features:
            num_evidences = len(evidence_global_features)
            prompt += f"\n\nYou have access to {num_evidences} relevant visual memory segments that may help answer the question."
        
        messages = [{"role": "user", "content": content}]
        
        # 处理输入
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # 生成回复
        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        answer = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        return answer



class Qwen2_5_VLWrapper:
    """Qwen2.5-VL包装器，与Qwen3VLWrapper使用相同的实现"""
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-2B-Instruct", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        初始化Qwen2.5-VL模型包装器
        Args:
            model_name: HuggingFace模型名称或路径
            device: 设备类型
        """
        self.device = device
        self.model_name = model_name
        
        # 根据模型名称选择正确的模型类
        model_class = _get_model_class(model_name)
        self.model = model_class.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map=device,
            trust_remote_code=True
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        
    @torch.no_grad()
    def encode_clip(self, frames: List[Image.Image]) -> Dict[str, torch.Tensor]:
        """
        输入: 一个滑动窗口内的若干帧 (list of PIL images)
        输出:
            - patch_tokens: (N_tokens, D)
            - global_feature: (D,)  # pooled token，可当作 video-CLS
        """
        # 处理多帧图像
        if not frames:
            raise ValueError("frames list cannot be empty")
        
        # 对于 Qwen2.5-VL，processor 需要文本输入
        # 检查是否是 Qwen2.5-VL processor（通过检查 processor 类型）
        is_qwen2_5 = "qwen2_5" in str(type(self.processor)).lower() or "qwen2.5" in str(type(self.processor)).lower()
        
        if is_qwen2_5:
            # Qwen2.5-VL 需要文本输入，使用一个简单的占位符文本
            # 使用列表格式，每个图像对应一个文本
            text_list = [""] * len(frames)
            inputs = self.processor(
                text=text_list,
                images=frames,
                return_tensors="pt",
                padding=True
            ).to(self.device)
        else:
            # Qwen2-VL 可以直接处理图像
            try:
                inputs = self.processor(
                    images=frames,
                    return_tensors="pt"
                ).to(self.device)
            except Exception as e:
                # 如果失败，尝试提供文本输入
                text_list = [""] * len(frames)
                inputs = self.processor(
                    text=text_list,
                    images=frames,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
        
        # 获取vision encoder输出
        # 兼容 Qwen2-VL 和 Qwen2.5-VL 的不同结构
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'visual'):
            vision_outputs = self.model.model.visual(**inputs['pixel_values'])
        elif hasattr(self.model, 'visual'):
            vision_outputs = self.model.visual(**inputs['pixel_values'])
        else:
            raise AttributeError(f"Cannot find vision encoder in model {self.model_name}")
        
        # 提取patch tokens和全局特征
        patch_tokens = vision_outputs.last_hidden_state  # (B, N_patches, D)
        
        # 全局特征：对patch tokens进行平均池化
        global_feature = patch_tokens.mean(dim=1).squeeze(0)  # (D,)
        
        # 展平patch tokens用于后续处理
        patch_tokens_flat = patch_tokens.view(-1, patch_tokens.size(-1))  # (B*N_patches, D)
        
        return {
            "patch_tokens": patch_tokens_flat,
            "global_feature": global_feature
        }

    @torch.no_grad()
    def update_explicit_memory(self, frames: List[Image.Image], current_memory_json: str) -> str:
        """
        使用 prompt + 当前 explicit map + 新视频片段，让 Qwen 输出新的 JSON。
        返回新的 explicit memory (string 格式，保持 JSON schema)
        """
        prompt = (
            "You are an indoor spatial map maintainer.\n"
            "Here is the current spatial relation map in JSON:\n"
            f"{current_memory_json}\n\n"
            "Now you observe a new egocentric video clip. "
            "Please update or correct the map according to this clip, "
            "and output ONLY the updated JSON with the same schema."
        )
        
        # 构建多模态输入
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": frame} for frame in frames
                ] + [{"type": "text", "text": prompt}]
            }
        ]
        
        # 处理vision信息
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # 生成回复
        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        # 提取JSON
        new_json_str = self._extract_json_from_text(output_text)
        return new_json_str

    def _extract_json_from_text(self, text: str) -> str:
        # 简单的提取逻辑：找到第一个 { 和 最后一个 }
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            return text[start:end]
        except ValueError:
            # 如果失败，就保留原来的（保守策略）
            return text

    @torch.no_grad()
    def get_text_embedding(self, question: str) -> torch.Tensor:
        """
        用 Qwen2.5-VL 的 text encoder / decoder hidden state 取一个问题的向量表示。
        training-free 检索备用。
        """
        # 使用processor处理文本
        inputs = self.processor(
            text=question,
            return_tensors="pt"
        ).to(self.device)
        
        # 获取文本embedding
        text_embeddings = self.model.model.embed_tokens(inputs.input_ids)  # (B, L, D)
        
        # 对序列维度求平均作为文本表示
        emb = text_embeddings.mean(dim=1).squeeze(0)  # (D,)
        return emb

    @torch.no_grad()
    def answer_with_evidence(
        self,
        question: str,
        explicit_memory_json: str,
        evidence_global_features: List[torch.Tensor]
    ) -> str:
        """
        使用：问题 + 显式记忆(JSON) + 若干检索到的 implicit memory tokens 进行最终推理。
        """
        prompt = (
            "You are an assistant that answers questions about an indoor environment.\n"
            "Here is the current spatial map in JSON:\n"
            f"{explicit_memory_json}\n\n"
            f"Question: {question}\n"
            "Please answer concisely."
        )
        
        # 构建消息
        content = [{"type": "text", "text": prompt}]
        
        # 如果有evidence features，可以尝试将其转换为图像表示
        if evidence_global_features:
            num_evidences = len(evidence_global_features)
            prompt += f"\n\nYou have access to {num_evidences} relevant visual memory segments that may help answer the question."
        
        messages = [{"role": "user", "content": content}]
        
        # 处理输入
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # 生成回复
        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        answer = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        return answer


class ReasoningModule:
    def __init__(self, qwen: Union[Qwen3VLWrapper, Qwen2_5_VLWrapper],
                 explicit_mem: 'ExplicitMemoryManager',
                 implicit_mem: 'ImplicitMemoryBank'):
        self.qwen = qwen
        self.explicit_mem = explicit_mem
        self.implicit_mem = implicit_mem

    @torch.no_grad()
    def retrieve_evidence_for_question(
        self,
        question: str,
        top_k: int = 5
    ) -> List['MemoryItem']:
        """
        零训练检索：用问题文本 embedding 与所有 implicit memory 做余弦相似度，
        选出 top-k 作为 evidence。
        """
        from .memory_module import MemoryItem
        
        if not self.implicit_mem.items:
            return []

        text_emb = self.qwen.get_text_embedding(question)        # (D,)
        feats = self.implicit_mem.as_matrix()                    # (N, D)
        # 归一化
        text_emb_n = text_emb / (text_emb.norm() + 1e-6)
        feats_n = feats / (feats.norm(dim=1, keepdim=True) + 1e-6)
        sims = (feats_n @ text_emb_n)                            # (N,)

        topk = min(top_k, sims.shape[0])
        values, indices = torch.topk(sims, k=topk, largest=True)
        selected_items = [self.implicit_mem.items[i.item()] for i in indices]
        return selected_items

    @torch.no_grad()
    def answer(self, question: str, top_k: int = 5) -> str:
        evidence_items = self.retrieve_evidence_for_question(question, top_k=top_k)
        evidence_feats = [it.global_feature for it in evidence_items]

        explicit_json = self.explicit_mem.to_json_str()
        answer = self.qwen.answer_with_evidence(
            question=question,
            explicit_memory_json=explicit_json,
            evidence_global_features=evidence_feats
        )
        return answer
