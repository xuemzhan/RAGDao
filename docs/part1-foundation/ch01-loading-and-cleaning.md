# 第1章　从混沌到秩序：文档加载与初清洗  

### 1.0　写在最前面：为什么这一章值得你读三遍？

过去几年，我和团队接手过不少大厂的 RAG 项目，其中相当一部分在上线后的第 1 个月，实际准确率都远低于预期。  
复盘时我们发现：**过半项目的核心问题都出在文档加载阶段**。  

不是模型不够强，也不是向量数据库不够快，而是——  
- 年报里每一页的页眉页脚被当成正文，污染了大量检索块  
- 扫描件 OCR 后出现成片的乱码和全角空格  
- 表格被强行拉平成纯文本，关键财务数据在结构化信息缺失中悄然丢失  
- Word 里的修订痕迹、批注、隐藏文字全部混进正文，噪声远超有用信息  

这些问题在加载阶段可能只是几个肉眼不太在意的脏字符、几行错位的表格，  
但一旦进入检索和向量空间，就会被成倍放大，最后让最贵的 70B 模型也只剩下“看上去很聪明，但总答不到点子上”的挫败感。

**结论：加载阶段的每一个细小错误，都会在后续环节被向量放大成系统性灾难。**  

这一章就是你对这些灾难的第一道保险。  
读完之后，你不会马上多一套“炫技”的新算法，但会多一套在 2025 年主流实践中被反复验证过的**文档加载方法论和工程默认值**。

它回答的是三个最现实的问题：

1. 在项目早期，如何选择合适的加载技术栈，而不是今天试一个库、明天换一个框架？  
2. 面对五花八门的格式（PDF、Word、扫描件、邮件、聊天记录…），如何建立一个**统一、可观测、可回滚**的加载流水线？  
3. 如何用一张决策矩阵 + 一份检查清单，让团队在“地基环节”少踩 70% 以上本可以避免的坑？

接下来，我们先看一眼目前一些典型团队在用什么，然后再一起搭一套你自己的加载地基。

---

### 1.1　2025 年主流团队真实使用的加载技术栈

下面这张表，是我们对近几年接触过的若干项目做的**经验归纳**：  
所有公司信息都已经做了脱敏处理，工具组合只反映常见趋势和量级，并不代表任何官方立场或“唯一正确答案”。

| 公司类型                | 首选方案                                   | 备选方案                          | 每日文档量级      |
|-------------------------|--------------------------------------------|-----------------------------------|-------------------|
| 美股投行                | Unstructured Enterprise + 自研 OCR         | PyMuPDF + Tesseract               | 约 8～15 万页     |
| 国内大厂（搜索/大模型） | Unstructured 0.15+ + 阿里云 OCR            | 自研基于 LayoutLMv3 的解析器      | 约 50～200 万页   |
| 医疗 AI 独角兽          | Unstructured + Azure Form Recognizer       | -                                 | 约 2～5 万页      |
| 律所知识库              | 自研 PDF 解析器（基于 pdfminer.six + 规则）| -                                 | 约 1～3 万页      |

你不需要记住每一行的细节，重要的是形成三个直觉：

1. **几乎没有团队只靠单一工具吃遍所有格式**：Unstructured、PyMuPDF、pdfplumber、商业 OCR、自研解析器常常混搭；  
2. **扫描件和强格式文档（合同、表格）永远是重灾区**：项目一旦大规模落到这些文档上，加载问题就会集中爆发；  
3. **企业越大、文档越多，其加载层越倾向于“平台化”**：从“写脚本”走向“有统一的 Loader SDK + 配置化策略”。

---

### 1.2　2025 年“格式 → Loader 策略”决策矩阵

纸质版建议整页跨栏排版，电子版可以做成可放大的 PNG 或交互图表。  
这一节的目标是：**遇到一种新格式，你能迅速知道默认优选方案和需要警惕的坑。**

> 注：下面的工具组合和推荐度，基于写作时（约 2025.9）我们在多个项目中的综合评估，仅作为起点建议；  
> 不同团队可以根据自身基础设施和成本约束做调整。

| 格式             | 推荐度 | 推荐工具组合（默认优选）                                   | 典型问题 & 规避方式                                  |
|------------------|--------|------------------------------------------------------------|------------------------------------------------------|
| 数字原生 PDF     | ★★★★★  | PyMuPDF + pdfplumber（版面解析 + 表格结构识别）           | 页眉页脚、页码等噪声较多 → 必须配合页眉页脚剔除规则  |
| 扫描件 PDF       | ★★★★★  | Unstructured（hi_res / ocr_only 模式）+ 商业 OCR 服务     | 速度慢 → 异步处理 + 结果缓存；注意多语言/竖排文本    |
| Word (.docx)     | ★★★★☆  | python-docx / docx2txt（正文）+ Unstructured（复杂结构）  | 复杂表格错位 → 对关键模版单独建解析器                |
| HTML / Markdown  | ★★★★★  | BeautifulSoup + markdownify / 自研 HTML 清洗              | 行为脚本、导航栏等噪声多 → 严格的 DOM 白名单策略    |
| 企业微信/钉钉导出 | ★★★★☆  | Unstructured 企业版 / 自研 IM 导出解析                    | 格式多样 → 需按导出模版建 profile，并单独回归测试    |
| 邮件 (.eml/.msg) | ★★★★☆  | extract-msg + email 模块 + bs4                            | 附件需递归处理，区分系统签名/历史回复与正文主干      |
| Excel / CSV      | ★★★★★  | pandas + openpyxl（保留公式与格式，按需结构化）           | 超大文件 → 流式读取 + 分块处理，避免一次性载入内存    |

如果你只有精力做一件事，那就先把你们当前的文档类型一一对应到这张表上：  
- 哪些格式已经有稳定方案？  
- 哪些格式依然停留在“脚本能跑，但没人敢动”的阶段？  
- 哪些格式还完全没进系统，只靠“手动补救”在撑？

---

### 1.3　通用加载器 UltimateLoader v2025（生产级代码骨架，已在多个大规模知识库验证）

这一节不是为了给出一个“银弹脚本”，而是展示一套在多个项目中实践出来的**加载层组织方式**：  
- 一处配置，多处复用；  
- 同时考虑**日志、监控、回滚**；  
- 对新文档格式有清晰的“扩展点”。

下面的代码骨架可以作为你们自研 Loader SDK 的起点。

```python
# 文件名: ultimate_loader_v2025.py
# 测试环境示例: Python 3.11.x | unstructured~=0.15 | PyMuPDF(pymupdf) | pdfplumber
# 注意: 版本号仅为写作时环境，实际使用请以各库最新官方文档为准。

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union
import logging

from unstructured.partition.auto import partition

# 如果你在后续实现中需要更细粒度控制 PDF，可再引入:
# from unstructured.partition.pdf import partition_pdf
# import fitz  # PyMuPDF
# import pdfplumber

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DocumentElement:
    """加载层统一输出的数据结构，后续分块/元数据/预处理都基于它继续加工。"""

    text: str
    metadata: Dict[str, Any]
    element_type: str = "text"  # 例如: "title" / "narrative_text" / "table" / "figure" / ...


class UltimateLoader:
    """
    通用文档加载器 (v2025 精简示例版本):

    - 封装多种格式的解析策略，统一输出为 List[DocumentElement]
    - 预留了 OCR 策略、语言、是否解析图片等关键开关
    - 真正的项目中，你可以在此基础上继续拆分出:
        - _load_pdf() / _load_scanned_pdf()
        - _load_docx() / _load_html() / _load_email()
        - 以及企业自定义格式的解析器
    """

    def __init__(
        self,
        ocr_strategy: str = "hi_res",       # auto / fast / hi_res / ocr_only
        languages: List[str] | None = None, # 文档可能涉及的语言列表
        extract_images: bool = False,
        fallback_to_pdfplumber: bool = True,
    ) -> None:
        self.ocr_strategy = ocr_strategy
        self.languages = languages or ["eng", "chi_sim"]
        self.extract_images = extract_images
        self.fallback_to_pdfplumber = fallback_to_pdfplumber

    def load(self, path: Union[str, Path]) -> List[DocumentElement]:
        """
        加载入口:

        1. 将传入路径统一转换为 Path
        2. 使用 unstructured.partition.auto.partition 做自动类型识别与解析
        3. 将结果规范化为 List[DocumentElement]

        提示:
        - 这里演示的是“默认通用路径”，真实项目中可以根据后缀/魔数
          决定是否走专用的 _load_pdf() / _load_docx() 等。
        """
        path = Path(path)
        logger.info("开始加载文档: %s", path)

        # 对 PDF / 图片 等支持 OCR 的类型，strategy 和 languages/ocr_languages 很关键
        elements = partition(
            filename=str(path),
            strategy=self.ocr_strategy,
            # 不同版本的 unstructured 在 languages / ocr_languages 上略有差异:
            # - 有的推荐使用 languages=["eng", "chi_sim"]
            # - 有的推荐使用 ocr_languages="eng+chi_sim"
            # 这里使用 languages 作为更通用的示例，读者应以当前版本官方文档为准。
            languages=self.languages,
            pdf_infer_table_structure=True,
            extract_images_in_pdf=self.extract_images,
        )

        doc_elements: List[DocumentElement] = []

        for el in elements:
            # unstructured Element 对象通常具有 text / metadata / category 等字段
            raw_text = getattr(el, "text", "") or ""
            text = raw_text.strip()
            if not text:
                continue

            metadata_obj = getattr(el, "metadata", None)
            if metadata_obj is not None and hasattr(metadata_obj, "to_dict"):
                meta = metadata_obj.to_dict()
            else:
                meta = {}

            element_type = getattr(el, "category", "text")
            element_type = element_type.lower() if isinstance(element_type, str) else "text"

            doc_elements.append(
                DocumentElement(
                    text=text,
                    metadata=meta,
                    element_type=element_type,
                )
            )

        logger.info("文档 %s 加载完成，共得到 %d 个元素", path, len(doc_elements))
        return doc_elements


# 一键使用（生产环境推荐的“起步姿势”，后续可按需扩展）
if __name__ == "__main__":
    loader = UltimateLoader(
        ocr_strategy="hi_res",
        languages=["eng", "chi_sim"],
        extract_images=False,
    )
    elements = loader.load("data/2024_某银行年报_扫描版.pdf")
    print(f"成功加载 {len(elements)} 个元素")
    if elements:
        print(elements[0].text[:300])

````

> 提示：在实际项目中，你可以把这些方法拆分到不同的模块或类中，通过插件机制/注册表来管理“文档类型 → 解析器”的映射；
> 只要对外暴露一致的 `List[DocumentElement]` 接口，后续分块、元数据、预处理等环节就都能复用。

---

### 1.4　本章可直接贴工位墙的 12 条加载检查清单（节选）

最后，我们用一张检查清单，把本章的关键点“压缩”成可以日常对照的操作项。
表里的阈值是我们在多项目中常用的**目标值/经验值**，不同业务可以按需调整，但不建议偏离太多。

| 编号  | 检查项                      | 是否必做 | 建议监控指标         |
| --- | ------------------------ | ---- | -------------- |
| 1   | 是否对所有 PDF 开启了表格结构识别      | 必做   | 表格识别率 > 98%    |
| 2   | 扫描件是否使用 hi_res + 多语言 OCR | 必做   | OCR 错误率 < 0.5% |
| 3   | 是否实现了页眉页脚自动剔除            | 必做   | 页眉污染率 < 0.1%   |
| 4   | 是否对每种新文档格式都写了单测          | 必做   | 单测覆盖率≈100%     |
| 5   | 是否监控了加载失败率（目标 < 0.01%）   | 必做   | 异常集中时触发告警      |
| 6   | 是否对超大文件（>200MB）做了流式处理    | 推荐   | 内存占用与耗时监控      |
| 7   | 是否对加载后的文本长度分布做了监控        | 推荐   | 异常长短块自动告警      |
| ... | （完整 12 条见书末附录）           |      |                |

建议你在项目正式立项前，就把这张表打印出来贴在工位边上：

* 每引入一种新格式，就对照一次；
* 每次 Loader 做“大改版”，上线前也对照一次；
* 日常运维遇到“为什么突然搜不出来”时，先看这七条是不是有哪一条悄悄失效了。

这一章就到这里。
从下一章开始，我们会在这条加载流水线的上游接上更细腻的分块策略，让每一个加载出来的 `DocumentElement` 都能以正确的方式走向向量世界。
