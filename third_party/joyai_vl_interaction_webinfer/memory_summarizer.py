# -*- coding: utf-8 -*-
import base64
import os
import re
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from PIL import Image
import io

# ============================================================
# Prompt Templates
# ============================================================

DETAILED_SUMMARY_PROMPT_EN = """\
You are writing mid-term memory for a long-running video agent. The user message contains timestamped key frames for Chunk {chunk_index}, covering {frame_range}; each image is preceded by its sampled timestamp span. These frames will be unavailable afterward, so your paragraph must preserve the key evidence that downstream models need for recall, reasoning, and answering follow-up questions once the visuals are gone.

[Output Format]
- Write a SINGLE factual paragraph{length_instruction}. Use the full output budget when the chunk is information-dense; for near-static or nearly duplicate frames, 1-2 brief sentences suffice — do not pad for length.
- When identity, text, or numeric values are uncertain, do not guess — mark them as "possibly X", "approx. X", or "unreadable"; even when unreadable, preserve the observable category, appearance, action, or positional features.

[Information to Preserve (by priority, highest first)]

Highest priority — essential for task continuity, must preserve:
- End-of-chunk handoff state: what remains on screen, what stage the task has reached, which objects are still available, which actions are unfinished or ongoing.
- Irreversible events and state changes: appearance, disappearance, movement, opening/closing, combination, separation, transfer of objects; source, destination, and resulting state of important operations.
- Intermediate results, task progress, partially completed or interrupted steps.
- Causal chains: when one action triggers another, keep the before/after relation instead of listing isolated fragments.

High priority — high information value and hard to reconstruct:
- Readable text, labels, numbers, measurements, counts, dates, identifiers, specifications; when multiple values appear together, preserve the "item → value" mapping instead of listing context-free numbers.
- Events, entities, or behaviors appearing for the first time or clearly anomalous within this chunk (judge only within this chunk, no cross-chunk history needed).
- Final positions of important objects and key spatial relations.

Medium priority — background and supporting detail:
- Inventory of people, body parts, tools, containers, ingredients, devices, and other entities present.
- Scene layout and persistent background: record once on first appearance, update on meaningful change, do not repeat if unchanged.
- Sparse time anchors for important events and transitions.

[Writing Rules]

Time references:
- Use at most 3-5 explicit time anchors in the paragraph, and only when they improve event separation or transition clarity. Prefer merged ranges of roughly {preferred_time_span}; use at most one explicit time range per continuous action phase. E.g., write "from 349s to 364s, the guitarist keeps playing, raises his left hand, then bows" instead of separate 349s-350s, 351s-352s, 353s-355s entries.
- Treat pre-image timestamp spans as evidence anchors, not sentence headers; do not copy each one into the output. Merge continuous or repeated actions across neighboring frames into one range and describe the visible changes within it.

Detail level and abstraction:
- Write in detail for highest and high priority information; keep stable background and repeated actions brief.
- Preserve concrete steps; do not replace them with vague wording like "preparation continues" or "performs related operations."
- Camera cuts, angle shifts, or viewpoint changes need not be mentioned individually; record them only when they reveal new elements, change readable information, or affect action continuity.
- When the scene contains structured information (tables, menus, forms, HUDs, subtitles, etc.), preserve field-value associations rather than merely summarizing the topic.
- This is memory, not a caption: preserve state, entities, actions, text, and numeric values in a balanced way rather than favoring only narrative, novelty, or motion.

Factual constraints:
- Use only details directly visible in the provided key frames; do not fill in off-screen content or speculate about future developments.
- Describe in temporal order of the key frames, but do not narrate every sampled interval individually.
- Avoid generic wrap-up phrases, meta commentary (e.g., "this chunk shows…"), and repeated sentence patterns.

Output ONLY the paragraph."""

DETAILED_SUMMARY_PROMPT = """\
你正在为一个长时间运行的视频智能体编写中期记忆。用户消息包含第 {chunk_index} 个片段的带时间戳关键帧，覆盖时间范围 {frame_range}；每张图片前标注其采样时间段。这些帧在本次之后不再可用，因此你的段落需要保留下游模型在没有视觉信息时进行回忆、推理和回答后续问题所需的关键证据。

【输出格式】
- 撰写一个事实性段落{length_instruction}。信息密集时充分利用空间；接近静止或几乎无新信息时（如长时间静止画面、近乎重复的帧），用 1-2 句简短记录即可，不要凑长度。
- 身份、文字、数值不确定时不要猜测，用"疑似 X"、"约 X"或"不可读"等方式标注；不可读时仍保留可观察到的类别、外观、动作或位置特征。
- 段落主体使用中文叙述；画面里能看到的英文（术语、人名、作品名、短语、年代标注等）每次出现都写成"中文译名（English 原文）"，不许只留一种，不许半中半英，画面没出现的英文不要补。

【需保留的信息（按优先级从高到低）】

最高优先级 — 任务连续性必需，必须保留：
- 片段结束时的交接状态：屏幕上剩余什么、任务推进到哪一步、哪些物体仍然可用、哪些动作未完成或仍在进行。
- 不可逆事件与状态变化：物体的出现、消失、移动、开合、组合、分离、转移；重要操作的来源、目的地与结果状态。
- 中间结果、任务进度、部分完成或被中断的步骤。
- 因果链：当一个动作引发另一个动作时，保持前后关系，不要列成孤立片段。

高优先级 — 信息价值高且难以重建：
- 可读的文字、标签、标题、数字、测量值、计数、日期、标识符、规格。多值同时出现时保留"项目→值"的映射，不要列出脱离上下文的数字。
- 在本片段内首次出现或明显异常的事件、实体或行为（仅按本片段判断，不需考虑跨片段历史）。
- 重要物体的最终位置与关键空间关系。

中优先级 — 背景与陪衬：
- 出现的人物、身体部位、工具、容器、食材、设备等实体清单。
- 场景布局与持续存在的背景：首次出现时记录一次，发生有意义变化时更新，未变化则不重复。
- 重要事件与转换的稀疏时间锚点。

【撰写规则】

时间引用：
- 整段最多使用 3-5 个显式时间锚点，且仅在能改善事件区分或转换识别时使用。优先采用约 {preferred_time_span} 的合并范围，一个连续动作阶段最多使用一个显式时间范围。例如写"从 349s 到 364s，吉他手持续弹奏、抬起左手并鞠躬"，而不是 349s-350s、351s-352s、353s-355s 分别写。
- 图片前的时间标注是证据锚点，不要逐个复制到输出中；相邻帧中重复或连续的动作合并为一个范围，并在其中描述可见变化。

详略与抽象层级：
- 详写最高与高优先级信息；略写稳定背景与重复动作。
- 保留具体步骤，不要用"准备继续"、"做相关操作"等空话替代可见动作。
- 镜头切换、机位移动或视角变化本身无需逐次提及；仅当它们揭示新元素、改变可读信息或影响动作连续性时才记录。
- 当场景包含结构化信息（表格、菜单、表单、HUD、字幕等）时，保留字段-值对应关系，而不仅概述主题。
- 这是记忆而非字幕：均衡保留状态、实体、动作、文字与数值，而不是只偏重叙事、新颖性或运动。

事实性约束：
- 仅使用所提供关键帧中直接可见的细节；不要补全画面外的内容或推测后续发展。
- 按关键帧的时间顺序描述，但不要逐一叙述每个采样间隔。
- 避免泛泛的总结性短语、元评论（如"本片段展示了……"）和重复句式。

仅输出该段落。"""

EMPTY_CHUNK_SUMMARY_TEMPLATE_EN = "No visually significant change is evident in frames {frame_range}."
EMPTY_CHUNK_SUMMARY_TEMPLATE = "在帧 {frame_range} 中未观察到明显的视觉变化。"

BATCH_COMPRESS_PROMPT_EN = """\
You are compressing multiple mid-term memory segments into long-term memory for a long-running video agent. The original frames for period {merged_range} will no longer be available, so the output must preserve the key evidence that downstream models need for recall, reasoning, and answering follow-up questions afterward.

MID-TERM SUMMARIES TO MERGE:
{summaries_text}

[Task Definition]
This is a compression task, not a simple merge. You must prioritize and discard to reduce information density rather than piling everything together. The input consists of N mid-term summaries, each with its own end-of-segment handoff state; after merging, only the **final segment's end state** matters to downstream — earlier segments' end states that have been superseded by later events (e.g., "a cup picked up then put down", "a menu opened then closed") should be treated as process information or omitted.

[Output Format]
- Write a SINGLE unified factual paragraph (NOT separate per-segment summaries), prioritizing within the target length{length_instruction}.
- Use the same language as the mid-term summaries; if languages are mixed across segments, use whichever is the majority.
- Uncertainty markers from mid-term summaries ("possibly X", "approx. X", "unreadable") must be carried through to long-term memory — do not "launder" them into definitive statements during merging.

[Retention Priority]

Highest priority — must preserve:
- The **final** handoff state at the end of the merged period: what remains on screen, what stage the task has reached, which objects are still available, which actions are unfinished or ongoing.
- Cross-segment irreversible events and state changes: final appearances/disappearances of objects, key movements, combinations, separations, transfers; source, destination, and resulting state of important operations.
- Readable text, labels, numbers, measurements, counts, dates, identifiers, specifications, and "item → value" mappings.
- Causal chains: when one action triggers a subsequent action, preserve the dependency instead of flattening into isolated facts.

High priority — should preserve:
- Entities (people, key tools, key objects) that persist across multiple segments and their evolution.
- Overall task progress structure and milestone stages.
- Events appearing for the first time or clearly anomalous within the merged range.
- Final positions of important objects and key spatial relations at the end of the merged period.

Compressible — actively reduce density or discard:
- Intermediate states that have been overwritten by later events — preserve only when the process itself carries information value.
- Stable background unchanged across segments — record once at first description only.
- Multiple similar repetitions of the same action — merge per the "Recurrence handling" rules below.
- Fine-grained procedural steps that can be omitted when the higher-level action is already recorded.

[Merging Rules]

Cross-segment consistency:
- When the same entity is named inconsistently across segments (e.g., "red cup" → "mug" → "container"), unify to the most informative and specific term, briefly noting alternative names on first occurrence.
- When multiple segments describe the same event with conflicting details, prefer the later or more specific version; if neither can be judged, keep both with "X or Y" rather than choosing one side.
- When the same event is described redundantly at adjacent segment boundaries, merge into a single record.

Recurrence handling:
- When the count of repetitions itself carries information (e.g., "knocked 5 times", "refreshed the page 3 times"), preserve the count.
- When repetition is procedural continuity (e.g., "keeps stirring", "repeatedly checks"), merge with one time range plus action description, e.g., "from 349s to 364s, keeps playing and raises hand to acknowledge the audience several times."
- Do not compress multiple distinct actions or state changes into a vague summary.

[Time References]
- Use 3-7 explicit time anchors across the paragraph (adjust for the span of {merged_range}), and only when they improve event separation or phase-transition clarity.
- Do not retain source summaries' <Xs-Ys> headers or reproduce their chain of "At <X>s..." timestamp cadence.
- Merge continuous or repeated neighboring actions into one range and describe the visible changes within it.

[Factual Constraints]
- Use only content already present in the mid-term summaries; do not fill in off-screen information or speculate about future developments.
- Preserve concrete steps and readable values; do not replace them with vague wording like "preparation continues" or "performs related operations."
- Avoid generic wrap-up phrases, meta commentary (e.g., "this memory covers…"), and repeated sentence patterns.

Output ONLY the unified narrative text, nothing else."""

BATCH_COMPRESS_PROMPT = """\
你正在为一个长时间运行的视频智能体将多段中期记忆压缩为长期记忆。时间段 {merged_range} 的原始帧将不再可用，因此输出需要保留下游模型在此之后进行回忆、推理和回答后续问题所需的关键证据。

需要合并的中期摘要：
{summaries_text}

【任务定位】
这是压缩任务，不是简单合并。需要按优先级取舍以降低信息密度，而不是把所有内容堆在一起。输入由 N 段中期摘要组成，每段都有自己的期末交接状态；合并后只有**最后一段的期末状态**对下游有意义，前面各段的期末状态若已被后续事件覆盖（如"拿起又放下的杯子"、"打开又关闭的菜单"），应作为过程信息处理或省略。

【输出格式】
- 撰写一个统一的事实性段落（不是按片段分开的摘要），在目标长度内{length_instruction}按优先级取舍。
- 中期摘要中的不确定性标注（"疑似 X"、"约 X"、"不可读"）必须传递到长期记忆，不要在合并时被"洗"成确定表述。

【保留优先级】

最高优先级 — 必须保留：
- 合并期**结束时**的最终交接状态：屏幕上剩余什么、任务推进到哪一步、哪些物体仍然可用、哪些动作未完成或仍在进行。
- 跨片段的不可逆事件与状态变化：物体的最终出现/消失、关键移动、组合、分离、转移；重要操作的来源、目的地与结果状态。
- 可读的文字、标签、数字、测量值、计数、日期、标识符、规格，以及"项目→值"的映射关系。
- 因果链：当一个动作引发后续动作时，保持依赖关系，不要扁平化为孤立事实。

高优先级 — 应当保留：
- 跨多个片段持续存在的实体（人物、关键工具、关键物体）及其演变。
- 任务的整体进度结构与阶段性里程碑。
- 在合并范围内首次出现或明显异常的事件。
- 重要物体在合并期末的位置与关键空间关系。

可压缩 — 主动降低密度或丢弃：
- 已被后续事件覆写的中间状态——仅在该过程本身有信息价值时才保留。
- 跨片段未变化的稳定背景，只在首次描述时记录一次。
- 同一动作的多次相似重复，按下方"重复处理"规则归并。
- 过程性的细微步骤，在更高层动作已被记录时可以省略。

【合并规则】

跨片段一致性：
- 同一实体在不同片段中名称不一致时（如"红色杯子" → "马克杯" → "容器"），统一为信息量最高、最具体的那个，并在首次出现时简要标注其他称呼。
- 多个片段对同一事件描述冲突时，优先采信时间靠后或信息更具体的版本；如果无法判定，用"X 或 Y"并列保留，不要单方面选边。
- 同一事件在相邻片段边界处被重复描述时，合并为一次记录。

重复处理：
- 当重复次数本身携带信息时（如"敲门 5 次"、"刷新页面 3 次"），保留次数。
- 当重复是过程性持续时（如"持续搅拌"、"反复检查"），用一个时间范围加动作描述合并，例如"从 349s 到 364s 持续弹奏并多次抬手致意"。
- 不要将多个不同的动作或状态变化压缩为模糊总结。

【时间引用】
- 整段使用 3-7 个显式时间锚点（按 {merged_range} 跨度调整），仅在能改善事件区分或阶段切换识别时使用。
- 不要保留源摘要的 <Xs-Ys> 标题，也不要复制源摘要中"在 <X>s..."的连续时间戳节奏。
- 连续或重复的相邻动作合并为一个范围，并在范围内描述可见变化。

【事实性约束】
- 仅使用中期摘要中已有的内容；不要补全画面外信息或推测后续发展。
- 保留具体步骤与可读数值，不要用"准备继续"、"做相关操作"等空话替代。
- 避免泛泛的总结性短语、元评论（如"这段记忆包含……"）和重复句式。

仅输出统一的叙述文本，不要输出其他内容。"""

# ============================================================
# Helper: encode image to base64 data URL
# ============================================================

def _encode_image_base64(image_path: str, max_pixels: int = 0) -> str:
    """Read an image file and return a base64 data URL for the OpenAI API.

    If *max_pixels* > 0 and the image exceeds that budget, it is
    down-scaled (preserving aspect ratio) so that width*height <= max_pixels.
    """
    if image_path.startswith("data:"):
        if max_pixels > 0:
            match = re.match(r"data:image/\w+;base64,(.+)", image_path)
            if match:
                img = Image.open(io.BytesIO(base64.b64decode(match.group(1))))
                w, h = img.size
                if w * h > max_pixels:
                    scale = (max_pixels / (w * h)) ** 0.5
                    new_w = max(1, int(w * scale))
                    new_h = max(1, int(h * scale))
                    img = img.resize((new_w, new_h), Image.LANCZOS)
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG")
                    data = base64.b64encode(buf.getvalue()).decode("utf-8")
                    return f"data:image/jpeg;base64,{data}"
        return image_path
    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    mime_type = mime_map.get(ext, "image/jpeg")

    if max_pixels > 0:
        img = Image.open(image_path)
        w, h = img.size
        if w * h > max_pixels:
            scale = (max_pixels / (w * h)) ** 0.5
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            img = img.resize((new_w, new_h), Image.LANCZOS)
            buf = io.BytesIO()
            save_fmt = "PNG" if ext == ".png" else "JPEG"
            img.save(buf, format=save_fmt)
            data = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:{mime_type};base64,{data}"

    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{data}"


# ============================================================
# SummarizerModel (OpenAI API client)
# ============================================================

class SummarizerModel:
    def __init__(
        self,
        model_name: str = "/tmp/models/Qwen3-VL-4B-Instruct",
        api_base: str = "http://localhost:8065/v1",
        longterm_model_name: str = "",
        longterm_api_base: str = "",
        mid_term_max_tokens: int = 5000,
        mid_term_target_tokens: int = 0,
        long_term_max_tokens: int = 1200,
        long_term_target_tokens: int = 0,
        key_frames_per_chunk: int = 8,
        max_pixels: int = 0,
        prompt_phase_seconds: float = 10.0,
        mid_term_temperature: float = 0.1,
        mid_term_top_p: float = 0.9,
        mid_term_top_k: int = -1,
        mid_term_repetition_penalty: float = 1.0,
        mid_term_presence_penalty: float = 0.0,
        long_term_temperature: float = 0.1,
        long_term_top_p: float = 0.9,
        long_term_top_k: int = -1,
        long_term_repetition_penalty: float = 1.0,
        long_term_presence_penalty: float = 0.0,
        debug: bool = False,
    ):
        self.model_name = model_name
        self.longterm_model_name = longterm_model_name or model_name
        self.mid_term_max_tokens = mid_term_max_tokens
        self.mid_term_target_tokens = mid_term_target_tokens
        self.long_term_max_tokens = long_term_max_tokens
        self.long_term_target_tokens = long_term_target_tokens
        self.key_frames_per_chunk = key_frames_per_chunk
        self.max_pixels = max_pixels
        self.prompt_phase_seconds = prompt_phase_seconds
        self.mid_term_temperature = mid_term_temperature
        self.mid_term_top_p = mid_term_top_p
        self.mid_term_top_k = mid_term_top_k
        self.mid_term_repetition_penalty = mid_term_repetition_penalty
        self.mid_term_presence_penalty = mid_term_presence_penalty
        self.long_term_temperature = long_term_temperature
        self.long_term_top_p = long_term_top_p
        self.long_term_top_k = long_term_top_k
        self.long_term_repetition_penalty = long_term_repetition_penalty
        self.long_term_presence_penalty = long_term_presence_penalty
        self.debug = debug

        # 中期摘要客户端（多模态，带图片）
        self._client = OpenAI(
            api_key="EMPTY",
            base_url=api_base,
        )

        # 长期压缩客户端（纯文本）—— 如果未指定则复用中期客户端
        if longterm_api_base and longterm_api_base != api_base:
            self._longterm_client = OpenAI(
                api_key="EMPTY",
                base_url=longterm_api_base,
            )
        else:
            self._longterm_client = self._client

        from transformers import AutoTokenizer
        self._tokenizer_cls = AutoTokenizer
        self._tokenizer_model_name = model_name
        self._tokenizer = None
        self._tokenizer_failed = False

    def preencode_frame(self, image_path: str) -> tuple[dict, float]:
        """Pre-encode one frame for mid-term summary use during streaming."""
        start_time = time.time()
        data_url = _encode_image_base64(image_path, max_pixels=self.max_pixels)
        return {
            "path": image_path,
            "data_url": data_url,
        }, time.time() - start_time

    def _chat(self, messages: list, max_tokens: int, temperature: float = 0.3, top_p: float = 0.9,
              top_k: int = -1, repetition_penalty: float = 1.0,
              presence_penalty: float = 0.0,
              client: OpenAI = None, model_name: str = None) -> str:
        """Call the vLLM OpenAI API server."""
        client = client or self._client
        model_name = model_name or self.model_name
        extra_body = {"greedy": False}
        if top_k > 0:
            extra_body["top_k"] = top_k
        if repetition_penalty != 1.0:
            extra_body["repetition_penalty"] = repetition_penalty
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            extra_body=extra_body,
        )
        return response.choices[0].message.content.strip() if response.choices else ""

    def _get_tokenizer(self):
        if self._tokenizer_failed:
            return None
        if self._tokenizer is None:
            try:
                self._tokenizer = self._tokenizer_cls.from_pretrained(
                    self._tokenizer_model_name
                )
            except Exception as e:
                print(f"WARNING: Failed to load tokenizer from {self._tokenizer_model_name}: {e}")
                self._tokenizer_failed = True
                return None
        return self._tokenizer

    def estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            return len(text) // 4
        return len(tokenizer.encode(text, add_special_tokens=False))

    def _parse_time_value(self, text: str):
        if text is None:
            return None
        value = text.strip()
        if value.endswith("s"):
            value = value[:-1]
        try:
            return float(value)
        except ValueError:
            return None

    @staticmethod
    def _split_frame_range(frame_range: str) -> tuple:
        """Split a frame_range string into (start, end).

        Handles both "0s-10s" and "0.0 seconds ~ 89.0 seconds" formats.
        """
        if " ~ " in frame_range:
            parts = frame_range.split(" ~ ", 1)
            return parts[0].strip(), parts[-1].strip()
        if "-" in frame_range:
            parts = frame_range.split("-", 1)
            return parts[0].strip(), parts[-1].strip()
        return frame_range.strip(), frame_range.strip()

    def _parse_time_range_bounds(self, time_range: str) -> tuple:
        if not time_range or "-" not in time_range:
            return None, None
        start_text, end_text = time_range.split("-", 1)
        return self._parse_time_value(start_text), self._parse_time_value(end_text)

    def _format_time_value(self, value: float) -> str:
        if value is None:
            return "0s"
        if abs(value - round(value)) < 1e-6:
            return f"{int(round(value))}s"
        return f"{value:.3f}".rstrip("0").rstrip(".") + "s"

    def _build_range_from_frame_indices(self, frame_time_ranges: list, start_idx: int, end_idx: int) -> str:
        start_value = float(start_idx)
        end_value = float(end_idx + 1)

        if start_idx < len(frame_time_ranges):
            parsed_start, _ = self._parse_time_range_bounds(frame_time_ranges[start_idx])
            if parsed_start is not None:
                start_value = parsed_start
        if end_idx < len(frame_time_ranges):
            _, parsed_end = self._parse_time_range_bounds(frame_time_ranges[end_idx])
            if parsed_end is not None:
                end_value = parsed_end

        if end_value <= start_value:
            end_value = start_value + 1.0

        return f"{self._format_time_value(start_value)}-{self._format_time_value(end_value)}"

    def _sample_sorted_indices(self, indices: list, budget: int) -> list:
        if budget <= 0 or not indices:
            return []
        if len(indices) <= budget:
            return list(indices)
        if budget == 1:
            return [indices[len(indices) // 2]]

        selected_positions = []
        last_pos = -1
        total = len(indices)
        for slot in range(budget):
            target = round(slot * (total - 1) / (budget - 1))
            target = max(target, last_pos + 1)
            max_allowed = total - (budget - slot)
            target = min(target, max_allowed)
            selected_positions.append(target)
            last_pos = target
        return [indices[pos] for pos in selected_positions]

    def _get_frame_prompt_time_range(self, frame: dict) -> str:
        return frame.get("source_time_range") or frame.get("time_range") or "unknown"

    def _build_prompt_phases(self, key_frames: list) -> list:
        return [
            {
                "time_range": self._get_frame_prompt_time_range(frame),
                "frames": [frame],
            }
            for frame in key_frames
        ]

    def _build_mid_term_debug_input(
        self,
        chunk_index: int,
        frame_range: str,
        frame_count: int,
        active_query: str,
        prompt_text: str,
        key_frames: list,
        phases: list,
        temperature: float,
    ) -> dict:
        content = [{"type": "text", "text": prompt_text}]
        for phase in phases:
            content.append({"type": "text", "text": f"<{phase['time_range']}>"})
            for frame in phase["frames"]:
                content.append({
                    "type": "image_path",
                    "image_path": frame["path"],
                    "max_pixels": self.max_pixels,
                })
        return {
            "stage": "mid_term",
            "model_name": self.model_name,
            "max_tokens": self.mid_term_max_tokens,
            "target_token_count": self.mid_term_target_tokens,
            "temperature": temperature,
            "top_p": self.mid_term_top_p,
            "chunk_index": chunk_index,
            "frame_range": frame_range,
            "frame_count": frame_count,
            "active_query": active_query or "(none)",
            "phase_seconds": self.prompt_phase_seconds,
            "phases": [
                {
                    "time_range": phase.get("time_range"),
                    "frame_count": len(phase.get("frames", [])),
                    "frame_indices": [frame.get("frame_index") for frame in phase.get("frames", [])],
                    "frame_time_ranges": [frame.get("time_range") for frame in phase.get("frames", [])],
                }
                for phase in phases
            ],
            "key_frames": [
                {
                    "time_range": frame.get("time_range"),
                    "source_time_range": frame.get("source_time_range"),
                    "path": frame.get("path"),
                    "frame_index": frame.get("frame_index"),
                    "range_start_index": frame.get("range_start_index"),
                    "range_end_index": frame.get("range_end_index"),
                }
                for frame in key_frames
            ],
            "messages": [{"role": "user", "content": content}],
            "prompt_text": prompt_text,
        }

    def _build_long_term_debug_input(
        self,
        merged_range: str,
        active_query: str,
        prompt_text: str,
        mid_term_summaries: list,
        temperature: float,
    ) -> dict:
        return {
            "stage": "long_term",
            "model_name": self.longterm_model_name,
            "max_tokens": self.long_term_max_tokens,
            "target_token_count": self.long_term_target_tokens,
            "temperature": temperature,
            "top_p": self.long_term_top_p,
            "merged_range": merged_range,
            "active_query": active_query or "(none)",
            "source_summaries": [
                {
                    "chunk_index": entry.get("chunk_index"),
                    "frame_range": entry.get("frame_range"),
                    "summary_text": entry.get("summary_text"),
                }
                for entry in mid_term_summaries
            ],
            "messages": [{"role": "user", "content": prompt_text}],
            "prompt_text": prompt_text,
        }

    def select_key_frames(
        self,
        image_paths: list,
        frame_time_ranges: list,
        key_frame_indices: list,
        cached_frame_payloads: list = None,
    ) -> list:
        """Select up to key_frames_per_chunk frames and carry cached payloads when available."""
        del key_frame_indices  # Mid-term summaries are now driven only by timestamped frames.

        n = len(image_paths)
        budget = self.key_frames_per_chunk
        if n == 0:
            return []

        if budget <= 0 or n <= budget:
            selected_indices = list(range(n))
        else:
            selected_indices = self._sample_sorted_indices(list(range(n)), budget)

        key_frames = []
        for pos, idx in enumerate(selected_indices):
            left_idx = 0 if pos == 0 else (selected_indices[pos - 1] + idx) // 2 + 1
            right_idx = n - 1 if pos == len(selected_indices) - 1 else (idx + selected_indices[pos + 1]) // 2
            left_idx = min(max(left_idx, 0), n - 1)
            right_idx = min(max(right_idx, left_idx), n - 1)

            cached_frame = None
            if cached_frame_payloads and idx < len(cached_frame_payloads):
                cached_frame = cached_frame_payloads[idx]

            path = image_paths[idx]
            if cached_frame and cached_frame.get("path"):
                path = cached_frame["path"]
            if not path.startswith("data:") and not os.path.exists(path):
                continue

            source_time_range = frame_time_ranges[idx] if idx < len(frame_time_ranges) else None
            key_frame = {
                "path": path,
                "source_time_range": source_time_range,
                "time_range": self._build_range_from_frame_indices(frame_time_ranges, left_idx, right_idx),
                "frame_index": idx,
                "range_start_index": left_idx,
                "range_end_index": right_idx,
            }
            if cached_frame and cached_frame.get("data_url"):
                key_frame["data_url"] = cached_frame["data_url"]
            key_frames.append(key_frame)
        return key_frames

    def _build_mid_term_length_instruction(self) -> str:
        if self.mid_term_target_tokens > 0:
            return f"，目标约 {self.mid_term_target_tokens} tokens"
        return ""

    def _build_long_term_length_instruction(self) -> str:
        if self.long_term_target_tokens > 0:
            return f"，目标约 {self.long_term_target_tokens} tokens"
        return ""

    def generate_detailed_summary(
        self,
        chunk_index: int,
        frame_range: str,
        key_frames: list,
        frame_count: int,
        active_query: str = "",
    ) -> tuple:
        """Generate a detailed multimodal summary for a chunk (Tier 2).

        Returns (summary_text, debug_input_or_None).
        """
        if not key_frames:
            return EMPTY_CHUNK_SUMMARY_TEMPLATE.format(frame_range=frame_range), None

        length_instruction = self._build_mid_term_length_instruction()
        mid_term_temperature = self.mid_term_temperature
        prompt_text = DETAILED_SUMMARY_PROMPT.format(
            frame_range=frame_range,
            chunk_index=chunk_index,
            length_instruction=length_instruction,
            preferred_time_span=f"{self.prompt_phase_seconds:g} seconds",
        )

        phases = self._build_prompt_phases(key_frames)

        debug_input = None
        if self.debug:
            debug_input = self._build_mid_term_debug_input(
                chunk_index=chunk_index,
                frame_range=frame_range,
                frame_count=frame_count,
                active_query=active_query,
                prompt_text=prompt_text,
                key_frames=key_frames,
                phases=phases,
                temperature=mid_term_temperature,
            )

        frames_to_encode = [
            frame for phase in phases for frame in phase["frames"]
            if not frame.get("data_url")
        ]
        if frames_to_encode:
            max_px = self.max_pixels
            with ThreadPoolExecutor(max_workers=min(len(frames_to_encode), 4)) as pool:
                futures = {
                    pool.submit(_encode_image_base64, f["path"], max_px): f
                    for f in frames_to_encode
                }
                for fut in as_completed(futures):
                    frame = futures[fut]
                    try:
                        frame["data_url"] = fut.result()
                    except Exception as e:
                        print(f"WARNING: Failed to encode image {frame['path']}: {e}")

        content_list = [{"type": "text", "text": prompt_text}]
        for phase in phases:
            content_list.append({"type": "text", "text": f"<{phase['time_range']}>"})
            for frame in phase["frames"]:
                data_url = frame.get("data_url")
                if data_url:
                    content_list.append({
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    })

        messages = [{"role": "user", "content": content_list}]
        summary_text = self._chat(
            messages,
            max_tokens=self.mid_term_max_tokens,
            temperature=mid_term_temperature,
            top_p=self.mid_term_top_p,
            top_k=self.mid_term_top_k,
            repetition_penalty=self.mid_term_repetition_penalty,
            presence_penalty=self.mid_term_presence_penalty,
        )
        return summary_text, debug_input

    def batch_compress_to_longterm(
        self,
        existing_longterm: str,
        mid_term_summaries: list,
    ) -> tuple:
        """
        Compress mid-term summaries and append to existing long-term memory.

        Only the new mid-term summaries are compressed (existing long-term is
        NOT re-compressed), then appended to the existing block.

        mid_term_summaries: list of dicts with keys 'frame_range', 'summary_text', etc.
        Returns (merged_text, token_count, compressed_new_text, debug_input_or_None).
        """

        if not mid_term_summaries:
            token_count = self.estimate_tokens(existing_longterm)
            return existing_longterm, token_count, "", None

        # 计算合并后的时间范围：取第一个 chunk 的起始和最后一个 chunk 的结束
        # frame_range 可能是 "0s-10s" 或 "0.0 seconds ~ 89.0 seconds" 两种格式
        first_range = mid_term_summaries[0]["frame_range"]
        last_range = mid_term_summaries[-1]["frame_range"]
        merged_start = self._split_frame_range(first_range)[0]
        merged_end = self._split_frame_range(last_range)[1]
        merged_range = f"{merged_start}-{merged_end}"

        # Step 1: Compress the new mid-term summaries into a single block
        summary_parts = []
        for entry in mid_term_summaries:
            summary_parts.append(
                f"<{entry['frame_range']}>\n{entry['summary_text']}"
            )
        summaries_text = "\n\n".join(summary_parts)

        active_query = ""
        for entry in mid_term_summaries:
            candidate_query = entry.get("query")
            if candidate_query:
                active_query = candidate_query
                break

        long_term_temperature = self.long_term_temperature
        length_instruction = self._build_long_term_length_instruction()
        prompt_text = BATCH_COMPRESS_PROMPT.format(
            summaries_text=summaries_text,
            length_instruction=length_instruction,
            merged_range=merged_range,
        )
        debug_input = None
        if self.debug:
            debug_input = self._build_long_term_debug_input(
                merged_range=merged_range,
                active_query=active_query,
                prompt_text=prompt_text,
                mid_term_summaries=mid_term_summaries,
                temperature=long_term_temperature,
            )

        compressed_new = self._chat(
            [{"role": "user", "content": prompt_text}],
            max_tokens=self.long_term_max_tokens,
            temperature=long_term_temperature,
            top_p=self.long_term_top_p,
            top_k=self.long_term_top_k,
            repetition_penalty=self.long_term_repetition_penalty,
            presence_penalty=self.long_term_presence_penalty,
            client=self._longterm_client,
            model_name=self.longterm_model_name,
        )
        # 在压缩结果前加上合并时间范围标记
        compressed_new = f"<{merged_range}>\n{compressed_new}"

        # Step 2: Append to existing long-term memory
        if existing_longterm:
            merged = existing_longterm.rstrip() + "\n\n" + compressed_new
        else:
            merged = compressed_new

        token_count = self.estimate_tokens(merged)
        return merged, token_count, compressed_new, debug_input

    def shutdown(self):
        """No-op: the vLLM server is managed externally."""
        pass
