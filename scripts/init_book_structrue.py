#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据预设的书本结构，初始化 docs 目录中的章节与附录结构。

使用方式：
    在项目根目录下（docs/ 与 scripts/ 同级）运行：
        python scripts/init_book_structure.py
"""

from pathlib import Path
import textwrap


def create_text_file(path: Path, content: str):
    """如果文本文件不存在则创建，并写入内容；存在则跳过。"""
    if path.exists():
        print(f"[跳过] 文件已存在: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"[创建] {path}")


def create_empty_file(path: Path):
    """创建一个空文件（用于占位，如图片等），若已存在则跳过。"""
    if path.exists():
        print(f"[跳过] 文件已存在: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    print(f"[创建空文件] {path}")


# ====================== 各文件内容模板 ======================

def readme_md() -> str:
    return textwrap.dedent("""\
    # （书籍封面 & 卷首语占位）

    > 提示：本文件作为整本书的首页（封面 + 卷首语）。
    > 你可以在这里放置标题、封面图、作者介绍、推荐语等。

    这里是书籍的封面与总卷首语区域，请根据实际项目进行替换和扩展。

    - 书名：TODO：填写你的书名（例如：《RAG 之道》）
    - 副标题：TODO
    - 作者：TODO
    - 简介：用几句话说明这本书解决什么问题、适合谁读

    接下来，你可以在左侧导航中，从 Part I 的卷首语与各章节开始阅读。
    """)


def summary_md() -> str:
    return textwrap.dedent("""\
    # Summary

    * [封面 & 卷首语](README.md)

    ## Part I　万丈高楼平地起 —— 数据准备与索引
    * [卷首语：我亲手埋葬过 63 个项目](part1-foundation/preface.md)
    * [第1章　从混沌到秩序：文档加载与初清洗](part1-foundation/ch01-loading-and-cleaning.md)
    * [第2章　分块的艺术：Chunking 策略全解析](part1-foundation/ch02-chunking-art.md)
    * [第3章　元数据富化：给每块内容安上“身份证”](part1-foundation/ch03-metadata-enrichment.md)
    * [第4章　文本预处理与标准化：嵌入前的最后一道关口](part1-foundation/ch04-text-preprocessing.md)
    * [Part I 结语 & 投资回报总览](part1-foundation/part1-summary.md)

    ## Part II　去伪存真，沙中淘金 —— 高级检索技术
    * （建设中）

    ## 附录
    * [术语表](appendices/glossary.md)
    * [参考文献](appendices/references.md)
    * [致谢](appendices/acknowledgements.md)
    """)


def part1_preface_md() -> str:
    return textwrap.dedent("""\
    # 卷首语：我亲手埋葬过 63 个项目…

    > 本卷首语用于为整本书，尤其是 Part I，定下基调：
    > 为什么“数据准备与索引”阶段，决定了后面一切效果的上限？

    这里可以讲：
    - 你在真实项目中的踩坑经历（尤其是因为数据准备不到位导致的失败）
    - 为什么大部分 RAG / 知识库项目死在「上线前」而不是模型本身
    - 本部分要解决的核心矛盾：从混沌文档到可被检索和理解的结构化知识

    （建议保留故事性 + 反思性 + 方法论的结合）
    """)


def part1_ch01_md() -> str:
    return textwrap.dedent("""\
    # 第1章　从混沌到秩序：文档加载与初清洗

    本章聚焦在 RAG 管道的起点：**如何把一堆“乱七八糟的文档”可靠地加载进来**。

    建议包含的内容示例：
    - 支持的文档类型（PDF、Word、PPT、网页、数据库等）
    - Loader 选择与实现策略（LangChain、LlamaIndex、自研 Loader 等）
    - 常见文档结构问题（页眉页脚、页码、目录、脚注、公式、表格等）
    - 初步清洗规则（删除噪声、合并断行、移除多余空白等）
    - 一个端到端示例：从原始文档到“初清洗后文本”的对比

    （后续可按实际项目经验进行细化与补充）
    """)


def part1_ch02_md() -> str:
    return textwrap.dedent("""\
    # 第2章　分块的艺术：Chunking 策略全解析

    本章讨论如何把长文本切成“既能理解上下文，又便于检索”的合理块。

    建议包含的内容示例：
    - 为什么必须分块？不分块会出现什么问题？
    - 常见分块策略：
      - 固定长度分块
      - 按段落 / 标题分块
      - 语义分块（基于模型 / 标点）
    - Chunk 大小与重叠（overlap）的经验值与调参思路
    - 面向不同场景的分块策略对比（FAQ、长报告、代码库、知识图谱等）
    - 一个可视化示例：同一文档在不同分块策略下的检索效果对比
    """)


def part1_ch03_md() -> str:
    return textwrap.dedent("""\
    # 第3章　元数据富化：给每块内容安上“身份证”

    元数据是让检索“有脑子”的关键，本章讲如何为每个 Chunk 设计和填充元数据。

    建议包含的内容示例：
    - 什么是元数据？为什么它对 RAG 至关重要？
    - 常见元数据字段：
      - 文档来源（source）
      - 章节 / 标题（section / heading）
      - 页码 / 段落号
      - 时间 / 版本号 / 作者
    - 面向业务场景的自定义元数据设计（如产品线、地区、客户类型等）
    - 如何在管道中统一管理元数据（schema 设计与追踪）
    - 元数据在过滤检索、排序、权限控制中的应用
    """)


def part1_ch04_md() -> str:
    return textwrap.dedent("""\
    # 第4章　文本预处理与标准化：嵌入前的最后一道关口

    在将文本送入嵌入模型之前，还有一系列“看似细碎但极其关键”的预处理步骤。

    建议包含的内容示例：
    - 基本文本规范化（大小写、标点、空白字符、特殊符号等）
    - 多语言 / 编码问题（UTF-8、全角半角、中英混排等）
    - 表格、代码块、公式的特殊处理策略
    - 去重与规整（重复内容、模板化内容等）
    - 对嵌入质量的健康检查方法与指标

    可以配合实测案例：预处理前后嵌入效果和检索相关性的变化。
    """)


def part1_summary_md() -> str:
    return textwrap.dedent("""\
    # Part I 结语：投资回报总览

    本部分建议：
    - 用 1～2 页，对 Part I 进行“复盘式”总结
    - 展示一张「端到端数据准备流程」的总览图（A3 大图思维导图风格）
    - 给出四章的“投入 vs 产出”对比表：
      - 如果忽略本章内容，项目常见后果？
      - 投入多少工程时间/成本，可以换来怎样的错误率下降 / 效果提升？

    可以考虑：
    - 给出一个“检查清单（Checklist）”：上线前确保已经做完的准备工作
    - 链接到后续章节（Part II 高级检索）的预告与发问
    """)


def appendices_glossary_md() -> str:
    return textwrap.dedent("""\
    # 附录 A：术语表（Glossary）

    > 收录全书中出现的核心术语，并给出**统一、简洁、可落地**的定义。

    示例结构：
    - **RAG（Retrieval-Augmented Generation）**：……
    - **Chunk / Chunking**：……
    - **Hybrid Retrieval（混合检索）**：……
    - **Rerank / 重排**：……
    - **Embedding / 向量化**：……

    建议：
    - 随着写作推进，持续补充与修订
    - 同一术语在不同章节保持一致用法
    """)


def appendices_references_md() -> str:
    return textwrap.dedent("""\
    # 附录 B：参考文献（References）

    可以包含：
    - 学术论文（按 APA / IEEE 等格式统一）
    - 官方文档链接（框架、库、云服务等）
    - 高质量博客 / 技术文章
    - 书籍

    建议：
    - 为每一章中引用的关键资料给出编号，并在正文中引用
    - 保持引用信息的完整性（作者、标题、年份、来源等）
    """)


def appendices_acknowledgements_md() -> str:
    return textwrap.dedent("""\
    # 附录 C：致谢（Acknowledgements）

    在这里感谢：
    - 合作的同事 / 朋友
    - 提供关键反馈的读者 / 社区
    - 公司 / 组织给予的资源与支持
    - 任何你认为值得特别感谢的人或机构

    致谢往往是读者感知“这本书背后的人和故事”的重要部分，
    可以适度真诚、稍微感性一些。
    """)


# ====================== 主逻辑 ======================

def main():
    # 找到项目根：scripts/ 的上一级目录
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    docs_root = project_root / "docs"

    print(f"[信息] 脚本位置: {script_path}")
    print(f"[信息] 项目根目录: {project_root}")
    print(f"[信息] docs 目录: {docs_root}")

    # 1. 确保 docs 目录存在
    docs_root.mkdir(parents=True, exist_ok=True)

    # 2. 顶层文件：README.md, SUMMARY.md
    create_text_file(docs_root / "README.md", readme_md())
    create_text_file(docs_root / "SUMMARY.md", summary_md())

    # 3. Part I 目录与文件
    part1_dir = docs_root / "part1-foundation"
    part1_assets_dir = part1_dir / "assets"
    part1_dir.mkdir(parents=True, exist_ok=True)
    part1_assets_dir.mkdir(parents=True, exist_ok=True)

    create_text_file(part1_dir / "preface.md", part1_preface_md())
    create_text_file(part1_dir / "ch01-loading-and-cleaning.md", part1_ch01_md())
    create_text_file(part1_dir / "ch02-chunking-art.md", part1_ch02_md())
    create_text_file(part1_dir / "ch03-metadata-enrichment.md", part1_ch03_md())
    create_text_file(part1_dir / "ch04-text-preprocessing.md", part1_ch04_md())
    create_text_file(part1_dir / "part1-summary.md", part1_summary_md())

    # Part I 资产占位图（空文件，后续可替换为真实图片）
    create_empty_file(part1_assets_dir / "part1-overview.png")
    create_empty_file(part1_assets_dir / "ch01-loader-matrix.png")
    create_empty_file(part1_assets_dir / "ch02-decision-tree.png")
    create_empty_file(part1_assets_dir / "ch03-metadata-pyramid.png")
    create_empty_file(part1_assets_dir / "ch04-cleanse-checklist.png")

    # 4. Part II ~ Part VIII 目录（先建空目录，后续按需填充）
    part2_dir = docs_root / "part2-retrieval"
    part2_dir.mkdir(parents=True, exist_ok=True)
    # 先创建占位章节文件，内容后续可细化
    for fname, title in [
        ("ch05-hybrid-retrieval.md", "第5章　混合检索（Hybrid Retrieval）"),
        ("ch06-rerank-3.0.md", "第6章　重排 3.0：从打分到决策"),
        ("ch07-query-rewriting.md", "第7章　Query Rewriting：教模型问出好问题"),
        ("ch08-multimodal-retrieval.md", "第8章　多模态检索：文本之外的世界"),
    ]:
        path = part2_dir / fname
        content = f"# {title}\n\n> 本章节内容建设中，占位文件，后续将补充完整论述与案例。\n"
        create_text_file(path, content)

    # 其余 Part 先建目录，内容后续再写
    for dirname, comment in [
        ("part3-prompt", "Part III　运筹帷幄，决胜千里 —— 提示工程（待建设）"),
        ("part4-vectordb", "Part IV　工欲善其事，必利其器 —— 向量数据库全攻略（待建设）"),
        ("part5-finetune", "Part V　精雕细琢，量体裁衣 —— RAG 专属微调（待建设）"),
        ("part6-efficiency", "Part VI　高效为王 —— 生产级 RAG 管道优化（待建设）"),
        ("part7-evaluation", "Part VII　度量与迭代 —— 评估与持续改进（待建设）"),
        ("part8-defense", "Part VIII　未雨绸缪 —— 极端情况处理与 AI 安全（待建设）"),
    ]:
        d = docs_root / dirname
        d.mkdir(parents=True, exist_ok=True)
        # 可选：给每个 Part 放一个占位 README
        readme_path = d / "README.md"
        readme_content = f"# {comment}\n\n> 本部分内容尚在筹备中，后续将补充章节与案例。\n"
        create_text_file(readme_path, readme_content)

    # 5. 附录 appendices
    appendices_dir = docs_root / "appendices"
    appendices_dir.mkdir(parents=True, exist_ok=True)

    create_text_file(appendices_dir / "glossary.md", appendices_glossary_md())
    create_text_file(appendices_dir / "references.md", appendices_references_md())
    create_text_file(appendices_dir / "acknowledgements.md", appendices_acknowledgements_md())

    print("\n[完成] docs 目录的书本章节结构已初始化/补全。")


if __name__ == "__main__":
    main()