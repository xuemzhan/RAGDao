好，我们直接进入「代码落地」阶段，把第 3 章里涉及的功能，全部用**优化后的示例代码**跟你前面定义好的占位符一一对上号。

我会按你在精修版正文里用到的占位符顺序来给：

* `[[CODE_BLOCK_3_3_METAFORGE_CORE_PY]]`
* `[[CODE_BLOCK_3_5_ENTITY_TAGGER_PY]]`
* `[[CODE_BLOCK_3_5_LANGUAGE_DETECT_PY]]`
* `[[CODE_BLOCK_3_6_BUSINESS_METADATA_ENRICHER_PY]]`
* `[[CODE_BLOCK_3_7_LLM_METADATA_GENERATOR_PY]]`
* `[[CODE_BLOCK_3_8_METADATA_SCHEMA_SQL]]`
* `[[CODE_BLOCK_3_9_METADATA_QUALITY_MONITOR_PY]]`

你可以**直接用下面代码替换原文中的占位符**。我在实现里有意识地吸收了原稿中这些片段的意图：

* `MetaForge` 类（spacy NER + summarizer + question generator）
* Qwen2 业务标签 prompt
* advanced_qa_prompt（多难度问题）
* metadata_vector_alignment_check（向量一致性巡检）
* Qdrant/Weaviate filter 示例（会在 3.8 里自然兼容）

---

## 1. MetaForge 核心：`[[CODE_BLOCK_3_3_METAFORGE_CORE_PY]]`

这是一个**骨架级统一入口**，不直接绑死到某个 NER/LLM 实现，而是把各处理器解耦成可插拔组件，方便你在工程里替换。

```python
# 文件名: meta_forge_v2025.py
# 说明:
# - 这是一个“元数据锻造厂”骨架示例，用于串联 L1~L4 各类处理器。
# - 真实项目中，建议把处理器拆到单独模块，便于单测与扩展。

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol, runtime_checkable
import logging

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """与前两章保持一致的基础块结构。

    在项目中可以直接复用第 1/2 章定义的 Chunk / DocumentElement，
    这里只保留 text + metadata 作为示例。
    """
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class MetadataProcessor(Protocol):
    """所有元数据处理器的统一接口约定。"""

    def __call__(self, chunk: Chunk) -> Chunk:  # pragma: no cover - 协议接口本身不测
        ...


class MetaForge:
    """MetaForge: 统一元数据富化流水线。

    设计要点:
    - processors 顺序执行，每个处理器负责一个“关注点”（L1/L2/L3/L4/生成型元数据等）
    - 处理器之间通过 Chunk.metadata 传递信息，避免相互强耦合
    """

    def __init__(self, processors: List[MetadataProcessor]) -> None:
        if not processors:
            raise ValueError("MetaForge 至少需要一个元数据处理器")
        self.processors = processors

    def enrich_chunk(self, chunk: Chunk) -> Chunk:
        """对单个 Chunk 进行元数据富化。"""
        for processor in self.processors:
            try:
                chunk = processor(chunk)
            except Exception as exc:  # 工程上建议细分异常
                logger.exception("MetaForge 处理器 %s 失败，跳过本处理器", processor)
        return chunk

    def enrich_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """对一批 Chunk 进行元数据富化。"""
        return [self.enrich_chunk(ch) for ch in chunks]
```

> 对应原稿：
>
> * 原来 `MetaForge` 里直接 new spacy / summarizer / question_generator。
> * 现在把它拆成「骨架 + 可插拔处理器」，spacy/LLM 逻辑放到后面的 `EntityTagger`、`LLMMetadataGenerator` 等类中。

---

## 2. 实体标注：`[[CODE_BLOCK_3_5_ENTITY_TAGGER_PY]]`

这是一个基于 spaCy 的中文（或多语种）实体抽取示例，和你原来 `zh_core_web_trf` 的思路对齐，但加了**延迟加载和容错**。

```python
# 文件名: entity_tagger_v2025.py
# 依赖:
#   pip install spacy
#   python -m spacy download zh_core_web_trf

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import logging

import spacy

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    text: str
    metadata: Dict[str, Any]


class EntityTagger:
    """基于 spaCy 的实体标注处理器（L3 内容元数据之一）。

    - 支持中文管线 (zh_core_web_trf)，需要提前下载模型
    - 也可以在初始化时替换为其他语言/多语言模型
    """

    def __init__(self, model_name: str = "zh_core_web_trf") -> None:
        self.model_name = model_name
        try:
            self.nlp = spacy.load(model_name)
        except OSError as exc:
            logger.error("加载 spaCy 模型失败: %s，实体标注功能将被禁用", model_name)
            self.nlp = None  # 允许系统在无 NER 能力下继续运行

    def __call__(self, chunk: Chunk) -> Chunk:
        if not self.nlp or not chunk.text.strip():
            return chunk

        doc = self.nlp(chunk.text)
        entities = []
        for ent in doc.ents:
            entities.append(
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                }
            )

        # 将实体写入 metadata.entities（追加而不是覆盖）
        meta = chunk.metadata.copy()
        old_entities = meta.get("entities", [])
        meta["entities"] = old_entities + entities
        chunk.metadata = meta
        return chunk
```

---

## 3. 语言检测：`[[CODE_BLOCK_3_5_LANGUAGE_DETECT_PY]]`

这里给一个用 `langdetect` 的最小实现，工程里你可以换成 CLD3 / fastText 等。

```python
# 文件名: language_detect_v2025.py
# 依赖:
#   pip install langdetect

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import logging

from langdetect import detect, DetectorFactory, LangDetectException

logger = logging.getLogger(__name__)
DetectorFactory.seed = 0  # 让结果更稳定


@dataclass
class Chunk:
    text: str
    metadata: Dict[str, Any]


class LangDetector:
    """为每个 Chunk 标注语言代码 (lang)。"""

    def __call__(self, chunk: Chunk) -> Chunk:
        text = (chunk.text or "").strip()
        if not text:
            return chunk

        try:
            lang = detect(text)
        except LangDetectException:
            lang = "unknown"

        meta = chunk.metadata.copy()
        meta.setdefault("lang", lang)
        chunk.metadata = meta
        return chunk
```

---

## 4. L4 业务元数据自动打标：`[[CODE_BLOCK_3_6_BUSINESS_METADATA_ENRICHER_PY]]`

这里把你原来那段 Qwen2 业务分类 prompt 落成一个**可替换 LLM 的处理器**。
我用伪接口 `llm_client.chat()` 表示，你可以在工程中换成 OpenAI / DashScope / 自家网关。

```python
# 文件名: business_metadata_enricher_v2025.py
# 说明:
# - 这是一个通用骨架，llm_client 需要由调用方注入
# - 重点展示: Prompt 设计 + JSON 解析 + 容错

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    text: str
    metadata: Dict[str, Any]


BUSINESS_CLASSIFIER_PROMPT = """
你是一个企业级知识库管理员。
根据以下文档片段，判断它最可能属于哪个部门、哪个产品线、哪个业务场景、哪个密级。
只需输出 JSON，严禁解释。

文档：
{chunk_text}

输出格式（必须是合法 JSON）：
{{
  "department": "财务部/法务部/人力资源部/技术部/销售部/其他 之一",
  "product_line": "企业信贷/财富管理/投资银行/其他 之一",
  "scenario": "合规审查/客户尽调/财报分析/合同审查/其他 之一",
  "confidentiality": "公开/内部/秘密/绝密 之一"
}}
""".strip()


class BusinessMetadataEnricher:
    """使用大模型为 Chunk 自动打 L4 业务元数据标签。"""

    def __init__(self, llm_client: Any, model_name: str) -> None:
        """
        参数:
            llm_client: 具备 chat 接口的客户端，需由调用方实现/注入
            model_name: 使用的模型名称（具体取决于你的推理服务）
        """
        self.llm_client = llm_client
        self.model_name = model_name

    def __call__(self, chunk: Chunk) -> Chunk:
        if not chunk.text.strip():
            return chunk

        prompt = BUSINESS_CLASSIFIER_PROMPT.format(chunk_text=chunk.text[:2000])

        try:
            # 伪代码: 根据你们的 LLM 网关调整
            resp = self.llm_client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一个严谨的企业知识库分类助手，只输出 JSON。"},
                    {"role": "user", "content": prompt},
                ],
            )
            content = resp["choices"][0]["message"]["content"]
        except Exception:
            logger.exception("调用业务分类 LLM 失败，跳过本次 L4 打标")
            return chunk

        try:
            biz_meta = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("业务分类 LLM 返回的内容非合法 JSON，内容: %s", content[:200])
            return chunk

        meta = chunk.metadata.copy()
        meta.update(
            {
                "department": biz_meta.get("department"),
                "product_line": biz_meta.get("product_line"),
                "scenario": biz_meta.get("scenario"),
                "confidentiality": biz_meta.get("confidentiality"),
            }
        )
        chunk.metadata = meta
        return chunk
```

---

## 5. 生成型元数据（summary + questions）：`[[CODE_BLOCK_3_7_LLM_METADATA_GENERATOR_PY]]`

这里综合了你原稿里的「Qwen2 summarizer + llama question_generator」和 `advanced_qa_prompt` 的思路，用统一的 chat 调用来做摘要+多难度问题生成。

```python
# 文件名: llm_metadata_generator_v2025.py
# 说明:
# - 使用单个 Chat-style 模型同时生成 summary + questions
# - prompt 中包含“5 个不同难度的问题”设计

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    text: str
    metadata: Dict[str, Any]


ADVANCED_QA_PROMPT = """
你是一个专业的知识库问答设计师。
请根据以下内容，先生成一个不超过 80 字的中文摘要，然后生成 5 个不同难度的问题（从易到难）：
1. 事实型（谁/什么/何时）
2. 理解型（为什么/如何）
3. 列表型（列出所有…）
4. 比较型（与…相比）
5. 推理型（如果…会怎样）

要求：
- 所有输出必须是合法 JSON
- 不要添加多余说明

内容：
{chunk_text}

输出格式示例：
{{
  "summary": "……",
  "questions": [
    "问题1",
    "问题2",
    "问题3",
    "问题4",
    "问题5"
  ]
}}
""".strip()


class LLMMetadataGenerator:
    """为 Chunk 生成摘要与多样化问题列表。"""

    def __init__(self, llm_client: Any, model_name: str) -> None:
        self.llm_client = llm_client
        self.model_name = model_name

    def __call__(self, chunk: Chunk) -> Chunk:
        if not chunk.text.strip():
            return chunk

        prompt = ADVANCED_QA_PROMPT.format(chunk_text=chunk.text[:2000])

        try:
            resp = self.llm_client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一个严谨的 JSON 生成助手，只输出 JSON。"},
                    {"role": "user", "content": prompt},
                ],
            )
            content = resp["choices"][0]["message"]["content"]
        except Exception:
            logger.exception("调用 LLM 生成 summary/questions 失败")
            return chunk

        try:
            obj = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("LLM 返回的 summary/questions 不是合法 JSON，内容: %s", content[:200])
            return chunk

        summary = (obj.get("summary") or "").strip()
        questions = [q.strip() for q in obj.get("questions", []) if isinstance(q, str) and q.strip()]

        meta = chunk.metadata.copy()
        if summary:
            # 强制控制摘要长度（例如 80 tokens 对应大约 160~200 中文字符）
            meta["summary"] = summary[:200]
        if questions:
            meta["suggested_questions"] = questions[:5]

        chunk.metadata = meta
        return chunk
```

---

## 6. 元数据表结构（PostgreSQL + pgvector）：`[[CODE_BLOCK_3_8_METADATA_SCHEMA_SQL]]`

这里给的是一个可以直接扔进 `psql` 的建表示例，涵盖 L1–L4 + 生成型元数据。
向量部分用 `pgvector` 的 `vector` 类型。

```sql
-- 文件名: metadata_schema_v2025.sql
-- 依赖:
--   CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
--   CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS kb_chunks (
    chunk_id        UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id       TEXT                NOT NULL,              -- L1
    source_type     TEXT                NOT NULL,
    file_name       TEXT,
    url             TEXT,
    doc_version     TEXT,
    ingest_at       TIMESTAMPTZ         NOT NULL DEFAULT NOW(),

    page_no         INTEGER,                                  -- L2
    heading_path    TEXT,
    element_type    TEXT,
    position_in_doc INTEGER,

    lang            VARCHAR(8),                               -- L3
    importance      REAL,
    entities        JSONB,                                    -- [{text, label, ...}]
    topics          TEXT[],

    tenant_id       TEXT,                                     -- L4
    org_id          TEXT,
    biz_domain      TEXT,
    risk_level      TEXT,
    acl             JSONB,                                    -- 权限信息，可选 {roles:[], users:[]}
    confidentiality TEXT,

    summary         TEXT,                                     -- 生成型元数据
    suggested_questions JSONB,                                -- ["问题1", "问题2", ...]
    keywords        TEXT[],

    embedding       VECTOR(1536),                             -- 视嵌入维度而定

    -- 一些常用索引（根据业务需要增删）
    CONSTRAINT kb_chunks_source_idx UNIQUE (source_id, position_in_doc)
);

CREATE INDEX IF NOT EXISTS idx_kb_chunks_source_id     ON kb_chunks (source_id);
CREATE INDEX IF NOT EXISTS idx_kb_chunks_tenant_id     ON kb_chunks (tenant_id);
CREATE INDEX IF NOT EXISTS idx_kb_chunks_biz_domain    ON kb_chunks (biz_domain);
CREATE INDEX IF NOT EXISTS idx_kb_chunks_risk_level    ON kb_chunks (risk_level);
CREATE INDEX IF NOT EXISTS idx_kb_chunks_lang          ON kb_chunks (lang);
CREATE INDEX IF NOT EXISTS idx_kb_chunks_element_type  ON kb_chunks (element_type);
CREATE INDEX IF NOT EXISTS idx_kb_chunks_confidentiality ON kb_chunks (confidentiality);

-- 向量相似度检索示例索引（可选）
CREATE INDEX IF NOT EXISTS idx_kb_chunks_embedding_ivfflat
ON kb_chunks USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

> 对应原稿中的：
>
> * Qdrant/Weaviate filter 示例可以很自然映射到这里的字段。
> * 你也可以在书里补一段「如何把 MetaQL 映射到具体存储（Postgres/Qdrant/Weaviate）」。

---

## 7. 元数据质量 & 一致性监控：`[[CODE_BLOCK_3_9_METADATA_QUALITY_MONITOR_PY]]`

最后这段把你原稿里的 `metadata_vector_alignment_check` 扩展成一个**小监控脚本**：
既检查 coverage，也做向量-文本一致性抽样。

```python
# 文件名: metadata_quality_monitor_v2025.py
# 说明:
# - 这是一个离线/定时任务脚本骨架，可挂在 Airflow/Cron 等调度系统
# - 假设你有:
#     - 一个 get_all_chunks() 接口，返回若干 {chunk_id, text, metadata, embedding}
#     - 一个 embedder.encode(texts) 接口，返回向量列表

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List
import logging
import math
from statistics import mean

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ChunkRecord:
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    embedding: List[float]  # 向量数据库中存储的 embedding


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError("向量维度不一致")
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def compute_coverage(chunks: Iterable[ChunkRecord], field: str) -> float:
    total = 0
    non_empty = 0
    for ch in chunks:
        total += 1
        value = ch.metadata.get(field)
        if value not in (None, "", [], {}):
            non_empty += 1
    return non_empty / total if total else math.nan


def metadata_vector_alignment_check(
    chunks: Iterable[ChunkRecord],
    embedder: Any,
    sample_size: int = 1000,
    threshold: float = 0.995,
) -> Dict[str, Any]:
    """抽样检查: 向量是否与文本+元数据一致（向量漂移监控）。

    参数:
        chunks:     全量 chunk 迭代器
        embedder:   具备 encode(List[str]) -> np.ndarray 的对象
        sample_size:抽样数量上限
        threshold:  相似度告警阈值
    """
    sampled: List[ChunkRecord] = []
    for i, ch in enumerate(chunks):
        if i >= sample_size:
            break
        sampled.append(ch)

    if not sampled:
        logger.warning("暂无可用于检测的 chunk 样本")
        return {}

    texts = [c.text for c in sampled]
    new_embeddings = embedder.encode(texts)  # 形如 (N, D) 的 numpy 数组

    sims: List[float] = []
    bad_cases: List[Dict[str, Any]] = []

    for rec, new_emb in zip(sampled, new_embeddings):
        stored = np.array(rec.embedding, dtype=float)
        sim = cosine_similarity(new_emb, stored)
        sims.append(sim)

        if sim < threshold:
            bad_cases.append(
                {
                    "chunk_id": rec.chunk_id,
                    "similarity": sim,
                }
            )

    avg_sim = mean(sims)
    min_sim = min(sims)

    logger.info(
        "元数据/向量一致性检查完成: avg_sim=%.4f, min_sim=%.4f, bad_cases=%d",
        avg_sim,
        min_sim,
        len(bad_cases),
    )

    return {
        "avg_similarity": avg_sim,
        "min_similarity": min_sim,
        "num_samples": len(sampled),
        "threshold": threshold,
        "num_bad_cases": len(bad_cases),
        "bad_cases": bad_cases,
    }
```

> 你可以在书里配一句伪代码示例：
>
> ```python
> chunks = db.fetch_latest_chunks()   # 返回若干 ChunkRecord
> summary = metadata_vector_alignment_check(chunks, embedder, sample_size=500)
> prometheus_client.push(summary)     # 或写入日志/监控系统
> ```
