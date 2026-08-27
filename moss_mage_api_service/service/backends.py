"""Checkpoint-native model adapters used by the HTTP service."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import patch

from PIL import Image


DEVICE_MAP_STRATEGIES = {"auto", "balanced", "balanced_low_0", "sequential"}


def resolve_device_map(strategy: str, device: str) -> str | dict[str, str]:
    """Resolve a named Accelerate strategy or explicit single-device placement."""
    strategy = str(strategy).strip().lower()
    if strategy == "single":
        return {"": device}
    if strategy not in DEVICE_MAP_STRATEGIES:
        choices = ", ".join(sorted(DEVICE_MAP_STRATEGIES | {"single"}))
        raise ValueError(f"Unknown device-map strategy {strategy!r}; choose one of: {choices}")
    return strategy


def model_cuda_devices(model: Any) -> list[str]:
    """Return the CUDA devices actually present in an Accelerate device map."""
    devices: set[str] = set()
    for value in getattr(model, "hf_device_map", {}).values():
        if isinstance(value, int):
            devices.add(f"cuda:{value}")
            continue
        text = str(value)
        if text.isdigit():
            devices.add(f"cuda:{text}")
        elif text.startswith("cuda"):
            devices.add(text)
    return sorted(devices, key=lambda item: int(item.split(":", 1)[1]))


def validate_balanced_placement(model: Any, strategy: str) -> list[str]:
    """Ensure the default strategy really consumes every visible CUDA device."""
    import torch

    devices = model_cuda_devices(model)
    visible_count = torch.cuda.device_count()
    if strategy == "balanced" and visible_count > 1 and len(devices) != visible_count:
        raise RuntimeError(
            "Balanced model placement did not use every visible GPU: "
            f"visible={visible_count}, used={devices}. Set MODEL_DEVICE_MAP=auto "
            "to allow partial placement or restrict CUDA_VISIBLE_DEVICES explicitly."
        )
    return devices


def install_mamba_transformers_compat() -> None:
    """Restore the Transformers 4 generation aliases used by mamba-ssm 2.2."""
    import transformers.generation as generation

    replacement = getattr(generation, "GenerateDecoderOnlyOutput", None)
    if replacement is None:
        return
    for name in ("GreedySearchDecoderOnlyOutput", "SampleDecoderOnlyOutput"):
        if not hasattr(generation, name):
            setattr(generation, name, replacement)


class MossVLBackend:
    """MOSS-VL adapter using checkpoint-provided offline and realtime methods."""

    def __init__(self, args: SimpleNamespace):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        print(f"Loading MOSS-VL from {args.model_path}", flush=True)
        self.processor = AutoProcessor.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            local_files_only=True,
            frame_extract_num_threads=1,
        )
        device_map = resolve_device_map(args.device_map, args.device)
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            local_files_only=True,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            attn_implementation=args.attn_implementation,
        ).eval()
        self.model_devices = validate_balanced_placement(self.model, args.device_map)
        print(
            f"MOSS-VL placement: strategy={args.device_map}, devices={self.model_devices}",
            flush=True,
        )

    def generate_query(self, query: dict[str, Any]) -> str:
        import torch

        with torch.inference_mode():
            result = self.model.offline_batch_generate(self.processor, [query])
        return str(result["results"][0]["text"]).strip()


class MageVLBackend:
    """Mage-VL adapter using the checkpoint processor and StreamMind model."""

    def __init__(self, args: SimpleNamespace):
        install_mamba_transformers_compat()
        from transformers import AutoModelForCausalLM, AutoProcessor

        print(f"Loading Mage-VL from {args.model_path}", flush=True)
        device_map = resolve_device_map(args.device_map, args.device)
        with patch("builtins.input", return_value="y"):
            self.processor = AutoProcessor.from_pretrained(
                args.model_path,
                trust_remote_code=True,
                local_files_only=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                trust_remote_code=True,
                local_files_only=True,
                device_map=device_map,
                torch_dtype="auto",
                attn_implementation=args.attn_implementation,
            ).eval()
        self.model_devices = validate_balanced_placement(self.model, args.device_map)
        print(
            f"Mage-VL placement: strategy={args.device_map}, devices={self.model_devices}",
            flush=True,
        )

    def _decode(self, output: Any, input_length: int) -> str:
        return self.processor.tokenizer.decode(
            output[0, input_length:], skip_special_tokens=True
        ).strip()

    def generate_media(
        self,
        prompt: str,
        *,
        video_path: Optional[str] = None,
        images: Optional[list[Image.Image]] = None,
        max_new_tokens: int = 128,
        num_frames: int = 32,
        max_pixels: int = 150000,
        codec: bool = True,
        temperature: float = 0.0,
        top_p: float = 0.9,
    ) -> str:
        import torch

        media_content = []
        if video_path:
            media_content.append({"type": "video"})
        media_content.extend({"type": "image"} for _ in (images or []))
        messages = [{
            "role": "user",
            "content": media_content + [{"type": "text", "text": prompt}],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if video_path and codec:
            inputs = self.processor(
                text=[text],
                videos=[video_path],
                images=images or None,
                video_backend="codec",
                max_pixels=max_pixels,
                codec_config={
                    "engine": "hevc",
                    "target_canvas": num_frames,
                    "patch": 16,
                },
                return_tensors="pt",
                padding=True,
            )
        elif video_path:
            inputs = self.processor(
                text=[text],
                videos=[video_path],
                images=images or None,
                num_frames=num_frames,
                return_tensors="pt",
                padding=True,
            )
        else:
            inputs = self.processor(
                text=[text], images=images or [], return_tensors="pt", padding=True
            )
        inputs = {
            key: (value.to(self.model.device) if hasattr(value, "to") else value)
            for key, value in inputs.items()
        }
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "repetition_penalty": 1.0,
        }
        if temperature > 0:
            generation_kwargs.update(temperature=temperature, top_p=top_p)
        with torch.inference_mode():
            output = self.model.generate(**inputs, **generation_kwargs)
        return self._decode(output, inputs["input_ids"].shape[1])
