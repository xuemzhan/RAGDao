好，直接进入“代码落地”环节，把第 4 章所有占位符一个个补齐。
你可以**逐块复制**回书稿中对应的 `[[CODE_BLOCK_4_x_...]]` 位置。

---

## 1. 通用文本标准化引擎：`[[CODE_BLOCK_4_3_CLEANSEMASTER_PY]]`

```python
# 文件名: cleanse_master_v2025.py
# 说明:
# - 通用文本清洗/标准化引擎骨架
# - 步骤包括: Unicode 归一化、控制字符剔除、空白标准化、全角/半角统一等
# - 真实工程中建议和第 1 章/第 3 章里的 Chunk / DocumentElement 结构打通

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Protocol, runtime_checkable, List
import logging
import re
import unicodedata

logger = logging.getLogger(__name__)


@dataclass
class TextRecord:
    """最小可用的文本记录结构。

    在项目中可以直接替换为:
    - 第 1 章的 DocumentElement
    - 第 2/3 章的 Chunk
    只要有 text + metadata 即可。
    """
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class CleanseStep(Protocol):
    """清洗步骤统一接口。"""

    def __call__(self, record: TextRecord) -> TextRecord:  # pragma: no cover
        ...


class UnicodeNormalizer:
    """Unicode 归一化 (默认 NFKC)。

    作用:
    - 合并等价表示 (如全角数字/字母、兼容字符)
    - 为后续规则减少“奇怪字符”的数量
    """

    def __init__(self, form: str = "NFKC") -> None:
        self.form = form

    def __call__(self, record: TextRecord) -> TextRecord:
        if not record.text:
            return record
        record.text = unicodedata.normalize(self.form, record.text)
        record.metadata.setdefault("cleanse", {})
        record.metadata["cleanse"]["unicode_normalized"] = self.form
        return record


class ControlCharStripper:
    """控制字符与明显乱码剔除。

    - 删除 C0/C1 控制字符 (U+0000 ~ U+001F, U+007F)
    - 保留正常换行 (\\n) 和制表符 (\\t) 可按需配置
    """

    CONTROL_CHAR_PATTERN = re.compile(
        r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]"
    )

    def __init__(self, keep_newline: bool = True, keep_tab: bool = True) -> None:
        self.keep_newline = keep_newline
        self.keep_tab = keep_tab

    def __call__(self, record: TextRecord) -> TextRecord:
        text = record.text
        if not text:
            return record

        # 临时替换要保留的字符
        placeholder_n = "\uFFF0"
        placeholder_t = "\uFFF1"

        if self.keep_newline:
            text = text.replace("\n", placeholder_n)
        if self.keep_tab:
            text = text.replace("\t", placeholder_t)

        # 删除控制字符
        text = self.CONTROL_CHAR_PATTERN.sub("", text)

        # 还原占位符
        if self.keep_newline:
            text = text.replace(placeholder_n, "\n")
        if self.keep_tab:
            text = text.replace(placeholder_t, "\t")

        record.text = text
        record.metadata.setdefault("cleanse", {})
        record.metadata["cleanse"]["control_chars_stripped"] = True
        return record


class WhitespaceNormalizer:
    """空白标准化。

    - 将连续空格压缩为单个空格
    - 将 Windows 风格换行 \\r\\n 统一为 \\n
    - 可根据需要配置是否保留段落间空行
    """

    def __init__(self, keep_empty_lines: bool = True) -> None:
        self.keep_empty_lines = keep_empty_lines

    def __call__(self, record: TextRecord) -> TextRecord:
        text = record.text
        if not text:
            return record

        # 统一换行风格
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        if self.keep_empty_lines:
            # 分段处理: 段内压缩空白，段与段之间保留一个空行
            lines = []
            for line in text.split("\n"):
                # 将制表符视为空格
                line = line.replace("\t", " ")
                # 连续空格压缩
                line = re.sub(r"[ ]{2,}", " ", line)
                lines.append(line.rstrip())
            text = "\n".join(lines)
        else:
            # 彻底压平，适合需要强制一行的场景
            text = text.replace("\t", " ")
            text = re.sub(r"\s+", " ", text).strip()

        record.text = text
        record.metadata.setdefault("cleanse", {})
        record.metadata["cleanse"]["whitespace_normalized"] = True
        return record


class FullwidthHalfwidthNormalizer:
    """全角/半角统一。

    - 基于 Unicode 归一化 + 简单启发式
    - 结合 UnicodeNormalizer 一起使用效果更佳
    """

    # 一些常见全角标点，如果归一化后仍未统一，可以在此补充映射
    EXTRA_MAP = {
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "：": ":",
        "；": ";",
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
    }

    def __call__(self, record: TextRecord) -> TextRecord:
        text = record.text
        if not text:
            return record

        text = "".join(self.EXTRA_MAP.get(ch, ch) for ch in text)
        record.text = text
        record.metadata.setdefault("cleanse", {})
        record.metadata["cleanse"]["fullwidth_halfwidth_normalized"] = True
        return record


class CleanseMaster:
    """CleanseMaster: 文本预处理与标准化主引擎。

    使用方式:
        steps = [
            UnicodeNormalizer("NFKC"),
            ControlCharStripper(),
            WhitespaceNormalizer(),
            FullwidthHalfwidthNormalizer(),
            # 此处可挂接领域/语言特定规则
        ]
        cm = CleanseMaster(steps)
        cleaned = cm.cleanse(record)
    """

    def __init__(self, steps: List[CleanseStep]) -> None:
        if not steps:
            raise ValueError("CleanseMaster 至少需要一个步骤")
        self.steps = steps

    def cleanse(self, record: TextRecord) -> TextRecord:
        for step in self.steps:
            try:
                record = step(record)
            except Exception:
                logger.exception("Cleanse 步骤 %s 失败，已跳过该步骤", step)
        return record

    def cleanse_batch(self, records: List[TextRecord]) -> List[TextRecord]:
        return [self.cleanse(r) for r in records]
```

---

## 2. 多语言清洗流程图：`[[CODE_BLOCK_4_4_MULTILINGUAL_CLEANSE_MERMAID]]`

```mermaid
flowchart LR
    A[原始文本 TextRecord] --> B[语言检测<br/>LangDetector]
    B --> C{是否多语言混排?}

    C -- 否 --> D[按单一语言路由<br/>选择对应 CleanseMaster 实例]
    C -- 是 --> E[按句子/段落切分<br/>对每段做语言检测]

    E --> F[多语言分段列表<br/>(segment, lang)]
    F --> G[对每个 segment 应用语言专用保护规则<br/>标记敏感模式]
    G --> H[对每个 segment 运行语言专用清洗逻辑]
    H --> I[恢复被保护内容<br/>合并为清洗后的文本]

    D --> J[输出单语言清洗结果]
    I --> J

    J --> K[输出新的 TextRecord<br/>(text + metadata.lang + cleanse_trace)]
```

---

## 3. 多语言清洗骨架：`[[CODE_BLOCK_4_4_MULTILINGUAL_CLEANSE_PY]]`

```python
# 文件名: multilingual_cleanse_v2025.py
# 说明:
# - 在 CleanseMaster 外包一层多语言分发逻辑
# - 简化版: 每条记录整体做语言检测，然后路由到对应 CleanseMaster
# - 需要更细粒度混排处理时，可在 TODO 处补充“按句/段分段”的实现

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional
import logging

from langdetect import DetectorFactory, detect, LangDetectException

from cleanse_master_v2025 import TextRecord, CleanseMaster  # 假设和前一段代码在同一工程

logger = logging.getLogger(__name__)
DetectorFactory.seed = 0  # langdetect 结果更稳定


@dataclass
class MultilingualConfig:
    """多语言清洗配置。

    - default_cleanser: 默认 CleanseMaster
    - per_lang_cleanser: 针对特定 language code 的 CleanseMaster
    """
    default_cleanser: CleanseMaster
    per_lang_cleanser: Mapping[str, CleanseMaster] = field(default_factory=dict)


class LangDetector:
    """简单的语言检测封装。"""

    def detect_lang(self, text: str) -> Optional[str]:
        text = (text or "").strip()
        if not text:
            return None
        try:
            return detect(text)
        except LangDetectException:
            return None


class MultilingualCleanser:
    """多语言感知的清洗入口。

    示例:
        zh_cleanser = CleanseMaster([...中文规则...])
        en_cleanser = CleanseMaster([...英文规则...])
        default_cleanser = CleanseMaster([...通用规则...])

        cfg = MultilingualConfig(
            default_cleanser=default_cleanser,
            per_lang_cleanser={"zh-cn": zh_cleanser, "zh": zh_cleanser, "en": en_cleanser},
        )
        ml_cleanser = MultilingualCleanser(cfg)
        cleaned = ml_cleanser.cleanse(record)
    """

    def __init__(self, config: MultilingualConfig) -> None:
        self.config = config
        self.lang_detector = LangDetector()

    def _route_cleanser(self, lang: Optional[str]) -> CleanseMaster:
        if lang:
            # langdetect 返回的可能是 'zh-cn' / 'en' / 'de' 等
            if lang in self.config.per_lang_cleanser:
                return self.config.per_lang_cleanser[lang]
            # 尝试只取主语言部分 'zh-cn' -> 'zh'
            main = lang.split("-")[0]
            if main in self.config.per_lang_cleanser:
                return self.config.per_lang_cleanser[main]
        return self.config.default_cleanser

    def cleanse(self, record: TextRecord) -> TextRecord:
        text = record.text
        if not text:
            return record

        lang = self.lang_detector.detect_lang(text)
        cleanser = self._route_cleanser(lang)
        cleaned = cleanser.cleanse(record)

        # 在 metadata 中标注语言信息
        cleaned.metadata.setdefault("lang_detect", {})
        cleaned.metadata["lang_detect"]["lang"] = lang or "unknown"
        cleaned.metadata["lang_detect"]["cleanser_used"] = (
            cleanser.__class__.__name__
        )
        return cleaned
```

> 真要做“句/段级混排拆分”，可以在 `cleanse()` 里：
>
> * 先用简单句法/换行拆分；
> * 对每段调用 `detect_lang`；
> * 对不同语言段分别 cleanse 后再合并。
>   书里保持骨架级实现就足够了。

---

## 4. 领域敏感正则表：`[[CODE_BLOCK_4_5_DOMAIN_REGEX_TABLE]]`

这里直接给一份 **Python 字典** 版本，方便在工程里复用（比如给保护器用）。

```python
# 文件名: domain_patterns_v2025.py
# 说明:
# - 常见领域敏感模式的正则模板合集
# - 可用于“先匹配/保护，后清洗，再恢复”策略

DOMAIN_PATTERNS: Dict[str, Dict[str, str]] = {
    "finance": {
        # 金额: 例如 1,234,567.89 元 / 1 234 567.89 CNY / 5.67 亿
        "amount": r"\b\d{1,3}(?:[, ]\d{3})*(?:\.\d+)?\s*(?:元|人民币|CNY|USD|EUR|亿|万)?",
        # 利率: 1.23% / 0.5 % / 年化 3.75%
        "interest_rate": r"(?:年化\s*)?\d+(?:\.\d+)?\s*%",
        # 合同/账号/订单编号 (示例)
        "contract_id": r"\b[A-Z]{2,5}-\d{4,10}\b",
        "account_no": r"\b\d{8,32}\b",
    },
    "legal": {
        # 条款编号: 第3.2条 / 条款 4.1.3 / Section 5.2
        "clause_no": r"(第\d+(\.\d+)*条)|(条款\s*\d+(\.\d+)*)|(Section\s+\d+(\.\d+)*)",
        # 法规/规范编号 (简化示例)
        "regulation": r"[A-Z]{2,5}/\d{2,4}-\d{2,4}",
    },
    "engineering": {
        # 信号/寄存器名称: SIGNAL_NAME / reg_control_01 / BMS.CellVoltage[3]
        "signal_name": r"\b[A-Za-z_][A-Za-z0-9_\.\[\]]{2,}\b",
        # 版本号: v1.0.3 / 2.3.4-beta
        "version": r"\bv?\d+\.\d+(?:\.\d+)?(?:-[a-zA-Z0-9]+)?\b",
        # IP / 端口 / CAN ID 等也可按需补充
    },
    "medical": {
        # 剂量: 5mg / 10 mg/kg / 0.5 μg/mL
        "dosage": r"\b\d+(?:\.\d+)?\s*(?:mg|mg/kg|μg/mL|g/L)\b",
        # 检验指标: ALT、AST 等，可按需扩充枚举表结合正则
    },
}
```

---

## 5. 多模态清洗流程图：`[[CODE_BLOCK_4_6_MULTIMODAL_CLEANSE_FLOW_MERMAID]]`

```mermaid
flowchart LR
    A[多模态元素流<br/>(type: text/table/figure/code/formula)] --> B{元素类型?}

    B -- text --> C[发送到 CleanseMaster / MultilingualCleanser<br/>执行通用文本清洗]
    B -- table --> D[保留表格结构<br/>对单元格文本做温和清洗<br/>保留数字/格式]
    B -- figure/image --> E[清理 OCR 文本噪声<br/>生成/清洗 alt text]
    B -- formula --> F[最小必要清洗 LaTeX/MathML<br/>保留原始表达<br/>可选生成自然语言解释]
    B -- code --> G[删除行号/多余注释噪声<br/>保留缩进/语言标识/文件上下文]

    C --> H[输出 cleaned_text_block]
    D --> H
    E --> H
    F --> H
    G --> H

    H --> I[写回到下游: 分块/元数据/索引<br/>供嵌入和检索使用]
```

---

## 6. 多模态清洗骨架：`[[CODE_BLOCK_4_6_MULTIMODAL_CLEANSE_PY]]`

```python
# 文件名: multimodal_cleanse_v2025.py
# 说明:
# - 统一处理 text/table/image/figure/code/formula 等多模态元素
# - 与第 2 章的多模态分块器、第 3 章的元数据设计相呼应

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from cleanse_master_v2025 import TextRecord, CleanseMaster


ElementType = Literal["text", "table", "figure", "image", "code", "formula"]


@dataclass
class MultiModalElement:
    element_type: ElementType
    text: str = ""                # 对于 table/image/formula/code，通常存放文本描述/OCR/源码
    payload: Dict[str, Any] = field(default_factory=dict)  # 结构化内容: 表格结构、图片 ID、代码片段等
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultimodalCleanser:
    """多模态清洗引擎骨架。

    - 对 text 直接调用 CleanseMaster
    - 对 table/figure/formula/code 采取“最小侵入式清洗”
    """

    def __init__(self, text_cleanser: CleanseMaster) -> None:
        self.text_cleanser = text_cleanser

    # --- 对外主入口 ---

    def cleanse_batch(self, elements: List[MultiModalElement]) -> List[MultiModalElement]:
        return [self.cleanse_one(el) for el in elements]

    def cleanse_one(self, element: MultiModalElement) -> MultiModalElement:
        etype = element.element_type

        if etype == "text":
            return self._cleanse_text(element)
        if etype == "table":
            return self._cleanse_table(element)
        if etype in ("figure", "image"):
            return self._cleanse_figure(element)
        if etype == "formula":
            return self._cleanse_formula(element)
        if etype == "code":
            return self._cleanse_code(element)

        # 未知类型直接返回
        return element

    # --- 各类型具体实现 ---

    def _cleanse_text(self, element: MultiModalElement) -> MultiModalElement:
        record = TextRecord(text=element.text, metadata=element.metadata)
        cleaned = self.text_cleanser.cleanse(record)
        element.text = cleaned.text
        element.metadata = cleaned.metadata
        return element

    def _cleanse_table(self, element: MultiModalElement) -> MultiModalElement:
        """对表格内容做温和清洗。

        payload 期望结构 (示例):
        {
            "cells": [[cell_text, ...], ...],
            "n_rows": ...,
            "n_cols": ...
        }
        """
        cells = element.payload.get("cells")
        if not cells:
            return element

        cleaned_cells: List[List[str]] = []
        for row in cells:
            new_row = []
            for cell in row:
                record = TextRecord(text=str(cell), metadata={})
                record = self.text_cleanser.cleanse(record)
                new_row.append(record.text)
            cleaned_cells.append(new_row)

        element.payload["cells"] = cleaned_cells
        return element

    def _cleanse_figure(self, element: MultiModalElement) -> MultiModalElement:
        """图像/图表的清洗。

        - 输入的 text 可以是 OCR 内容或 alt text
        - 这里只做基本清洗，避免破坏有用信息
        """
        if not element.text:
            return element

        record = TextRecord(text=element.text, metadata=element.metadata)
        cleaned = self.text_cleanser.cleanse(record)
        element.text = cleaned.text
        element.metadata = cleaned.metadata
        return element

    def _cleanse_formula(self, element: MultiModalElement) -> MultiModalElement:
        """公式清洗。

        - 对 LaTeX/MathML 做最小必要整理
        - 不应改变任何数学含义
        """
        tex = element.text or ""
        if not tex:
            return element

        # 示例: 简单去掉连续多余空白，但保留换行
        tex = tex.replace("\r\n", "\n").replace("\r", "\n")
        tex_lines = [line.rstrip() for line in tex.split("\n")]
        tex = "\n".join(tex_lines)

        element.text = tex
        # 可在 metadata 中记录已处理的标记
        element.metadata.setdefault("cleanse", {})
        element.metadata["cleanse"]["formula_trimmed"] = True
        return element

    def _cleanse_code(self, element: MultiModalElement) -> MultiModalElement:
        """代码块清洗。

        - 删除无意义的行号前缀 (如 "1  int main()")
        - 删除复制时引入的提示信息
        - 保留缩进和代码结构
        """
        code = element.text or ""
        if not code:
            return element

        lines = code.splitlines()
        cleaned_lines: List[str] = []

        for line in lines:
            # 删除行首形如 "12  " 的行号
            new_line = re.sub(r"^\s*\d+\s+", "", line)
            cleaned_lines.append(new_line.rstrip())

        element.text = "\n".join(cleaned_lines)
        element.metadata.setdefault("cleanse", {})
        element.metadata["cleanse"]["code_lineno_stripped"] = True
        return element
```

---

## 7. 文本清洗效果监控脚本：`[[CODE_BLOCK_4_7_PREPROCESS_QUALITY_MONITOR_PY]]`

```python
# 文件名: preprocess_quality_monitor_v2025.py
# 说明:
# - 对清洗前/后的 TextRecord 做长度分布 & 字符类别分布对比
# - 抽取差异最大的若干样本，便于人工审查

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple
import collections
import json
import logging
import math
import statistics
import unicodedata

logger = logging.getLogger(__name__)


@dataclass
class TextRecordSnapshot:
    record_id: str
    text_before: str
    text_after: str
    metadata_before: Dict[str, Any]
    metadata_after: Dict[str, Any]


def char_category(ch: str) -> str:
    """根据 Unicode 类别粗分字符类型。"""
    if ch.isspace():
        return "whitespace"
    if ch.isdigit():
        return "digit"
    if ch.isalpha():
        return "letter"
    cat = unicodedata.category(ch)
    if cat.startswith("P"):
        return "punct"
    return "other"


def length_stats(values: List[int]) -> Dict[str, float]:
    if not values:
        return {"min": math.nan, "max": math.nan, "mean": math.nan, "p50": math.nan, "p90": math.nan}
    values_sorted = sorted(values)
    n = len(values_sorted)
    return {
        "min": float(values_sorted[0]),
        "max": float(values_sorted[-1]),
        "mean": float(statistics.mean(values_sorted)),
        "p50": float(values_sorted[n // 2]),
        "p90": float(values_sorted[int(n * 0.9)]),
    }


def char_distribution(texts: Iterable[str]) -> Dict[str, float]:
    counter = collections.Counter()
    total = 0
    for t in texts:
        for ch in t:
            cat = char_category(ch)
            counter[cat] += 1
            total += 1
    if total == 0:
        return {}
    return {k: v / total for k, v in counter.items()}


def compute_diffs(snapshots: List[TextRecordSnapshot]) -> Dict[str, Any]:
    lengths_before = [len(s.text_before) for s in snapshots]
    lengths_after = [len(s.text_after) for s in snapshots]

    len_stats_before = length_stats(lengths_before)
    len_stats_after = length_stats(lengths_after)

    dist_before = char_distribution(s.text_before for s in snapshots)
    dist_after = char_distribution(s.text_after for s in snapshots)

    # 找出长度变化最大的若干样本
    deltas: List[Tuple[str, int]] = []
    for s in snapshots:
        deltas.append((s.record_id, len(s.text_after) - len(s.text_before)))
    deltas_sorted = sorted(deltas, key=lambda x: abs(x[1]), reverse=True)
    top_changed = deltas_sorted[:20]

    return {
        "length_stats_before": len_stats_before,
        "length_stats_after": len_stats_after,
        "char_dist_before": dist_before,
        "char_dist_after": dist_after,
        "top_changed_records": [
            {"record_id": rid, "len_delta": delta} for rid, delta in top_changed
        ],
    }


def generate_report(snapshots: List[TextRecordSnapshot]) -> str:
    """生成 JSON 报告字符串，可写入文件或推送到监控系统。"""
    summary = compute_diffs(snapshots)
    return json.dumps(summary, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # 示例: 假设你已经在清洗流水线中记录了前/后的快照
    examples = [
        TextRecordSnapshot(
            record_id="chunk-001",
            text_before="2024 年净利润　5.67　亿元",
            text_after="2024 年净利润 5.67 亿元",
            metadata_before={},
            metadata_after={},
        ),
        TextRecordSnapshot(
            record_id="chunk-002",
            text_before="Line1\r\nLine2\r\n",
            text_after="Line1\nLine2",
            metadata_before={},
            metadata_after={},
        ),
    ]

    report = generate_report(examples)
    print(report)
```

---

如果你后面准备做一个**Part I 的配套代码仓库**，这几章的示例几乎可以直接按模块/包结构整理进去：

* `loaders/`（第 1 章）
* `chunking/`（第 2 章）
* `metadata/`（第 3 章）
* `preprocess/`（第 4 章：cleanse_master_v2025、多语言、多模态、质量监控）

后面如果你希望，我也可以帮你：

* 设计一个 **统一的 Python 包结构 + `pyproject.toml`/`setup.cfg`**；
* 把这四章的代码骨架打包成一个 `ragdao-foundation` 小库，让读者可以 `pip install` 后直接跑书里的示例。
