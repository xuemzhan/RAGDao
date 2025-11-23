# A3：元数据丰富 (Metadata Enrichment)——赋予数据可过滤的“维度”

元数据（Metadata）是“关于数据的数据”。元数据是提升检索精度的重要补充。许多向量数据库支持在进行向量搜索前，先通过元数据对搜索空间进行预过滤（Pre-filtering），这能极大地缩小匹配范围，获得更精准、更快速的检索结果。在RAG中，为每个文本块附加元数据，就如同为图书馆里的每一本书都制作一张详细的**索引卡**，上面不仅有书名，还有作者、出版日期、类别、关键词等。这使得我们不仅可以根据书的内容（语义搜索），还可以根据索引卡上的这些标签来进行精确的筛选和查找（元数据过滤）。

## **源元数据 (Source Metadata)**

一个生产级的RAG系统，其元数据应至少包含以下内容，形成一个结构化的“**内容身份证**”：

- **a. 来源与时间信息:**
    - source: 文档来源的URL或文件路径。
    - created_at, last_modified: 创建和最后修改时间戳。
- **b. 权责与版本信息:**
    - author: 文档作者。
    - version: 文档版本号，对于处理冲突和更新至关重要。
- **c. 内容分类信息:**
    - category: 主题分类或标签。
    - keywords: 人工或自动提取的关键词。
- **d. [新增] 结构化信息:**
    - doc_structure: 块在文档中的逻辑位置，如章节、子章节标题。
    - position: 块在段落中的物理位置（如页码、段落号）。
- **e. [新增] 实体识别结果:**
    - entities: 自动识别出的人名、地名、组织机构名等。

### **详细阐述：** 这是指直接来源于文档本身或其环境的描述性信息。

- **例子：**
    - source: 文件名或URL (/path/to/my_doc.pdf, https://en.wikipedia.org/wiki/RAG)
    - created_at, last_modified: 文件创建或最后修改日期
    - author: 文档作者
    - doc_type: 文档类型 (PDF, Markdown, API_Doc)
    - page_number: 对于PDF等文档，块所在的页码
    - chapter_title: 块所属的章节标题

## **内容元数据 (Content-based Metadata)**

- **详细阐述：** 这是通过自动化流程从块的**内容本身**提取出的结构化信息，极大地增强了数据的可发现性。
- **方法与示例：**
    - **实体提取 (Entity Extraction):** 使用命名实体识别（NER）模型（如spaCy）来自动识别和标注文本中的人名、地名、组织、产品名等。
    - **摘要 (Summarization):** 使用一个小型LLM为每个块生成一个简短的摘要。
    - **生成问题 (Question Generation):** 使用LLM，让它针对每个块生成几个最可能被问到的问题。
- **可执行代码示例 (使用spaCy进行实体提取):**
    
    ```python
    # 准备环境 (spaCy已在A1中安装和下载)
    import spacy
    
    # 加载spaCy模型 (如果之前没加载)
    try:
        nlp
    except NameError:
        print("Loading spaCy model for NER...")
        nlp = spacy.load("en_core_web_sm")
        print("Model loaded.")
    
    text_chunk = "Apple Inc., led by Tim Cook, is planning to build a new campus in Austin, Texas for $1 billion."
    doc = nlp(text_chunk)
    
    # 从处理后的文档中提取实体
    # ent.text 是实体文本, ent.label_ 是实体类型
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    
    # 将这些实体作为元数据
    chunk_with_metadata = {
        "text_content": text_chunk,
        "metadata": {
            "source": "press_release_q4_2024.txt",
            "entities": entities
        }
    }
    
    import json
    print(json.dumps(chunk_with_metadata, indent=2))
    # 输出:
    # {
    #   "text_content": "Apple Inc., led by Tim Cook, is planning to build a new campus in Austin, Texas for $1 billion.",
    #   "metadata": {
    #     "source": "press_release_q4_2024.txt",
    #     "entities": [
    #       { "text": "Apple Inc.", "label": "ORG" },
    #       { "text": "Tim Cook", "label": "PERSON" },
    #       { "text": "Austin", "label": "GPE" },
    #       { "text": "Texas", "label": "GPE" },
    #       { "text": "$1 billion", "label": "MONEY" }
    #     ]
    #   }
    # }
    ```