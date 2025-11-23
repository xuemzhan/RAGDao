# G1：综合评估指标 (Comprehensive Evaluation Metrics)——RAG系统的“体检报告”

对RAG系统的评估是一个多维度的任务，因为它本质上由两个核心组件构成：**检索器（Retriever）和生成器（Generator）**。因此，我们的评估指标也必须能分别（或综合）地考察这两个组件的性能。这就像给一个**学生**做一次全面的**体检**。

- **检索质量指标** 就像是检查他的**“视力”**。他的眼睛（检索器）能否在书本中快速、准确地找到知识点？
- **生成质量指标** 就像是检查他的**“大脑和口才”**。他（生成器）能否基于找到的知识点，进行清晰、准确、有逻辑的论述？

## **检索质量指标 (Retrieval Quality Metrics)**

- **详细阐述：** 这部分指标专注于评估检索器找回的上下文（Context）的质量。
- **核心指标：**
    - **上下文精确率 (Context Precision):** 找回的文档中，有多少是真正相关的？（视力是否清晰，杂光少？）
    - **上下文召回率 (Context Recall):** 所有应该被找回的相关文档，我们找回了多少？（视野是否开阔，不漏掉东西？）

## **生成质量指标 (Generation Quality Metrics)**

- **详细阐述：** 这部分指标评估LLM在给定上下文后，生成的答案本身的质量。
- **核心指标 (以RAGAS框架为代表):**
    - **忠实度 (Faithfulness):** 答案是否完全基于所提供的上下文？有没有“说谎”或“幻觉”？（是否诚实？）
    - **答案相关性 (Answer Relevancy):** 答案是否直接、有效地回应了用户的原始问题？（是否跑题？）
    - **答案正确性 (Answer Correctness):** 答案的内容是否事实准确？（知识是否正确？）

## **端到端自动化评估框架**

- **RAGAS (Retrieval-Augmented Generation Assessment):**
    - **概念出处：** RAGAS由Exploding Gradients、Langchain和Upstage AI的研究人员于2023年提出，论文为 "RAGAS: Automated Evaluation of Retrieval Augmented Generation"。它已迅速成为RAG评估领域的标准框架之一。
    - **工作原理：** 巧妙地利用强大的LLM（如GPT-3.5/4）自身作为评估者（裁判），来自动化地计算上述大部分指标。
- **可执行代码示例 (使用RAGAS进行评估):**

```python
# 准备环境:
# pip install ragas datasets openai
# 请确保你已经设置了OpenAI的API Key环境变量:
# export OPENAI_API_KEY='sk-...'

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
import os

# 检查API密钥是否存在
if "OPENAI_API_KEY" not in os.environ:
    print("ERROR: OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
else:
    # 1. 准备评估数据集
    #    这是评估的核心，你需要准备一个包含以下字段的测试集：
    #    - question: 用户的提问
    #    - answer: 你的RAG系统生成的答案
    #    - contexts: 你的RAG系统为生成该答案而检索到的文档块列表
    #    - ground_truth: (用于评估答案正确性) 一个由人工编写的、标准的黄金答案
    data_samples = {
        'question': [
            "What is the capital of France?", 
            "Who wrote 'To Kill a Mockingbird'?"
        ],
        'answer': [
            "Paris is the capital of France.", 
            "The author of 'To Kill a Mockingbird' is Harper Lee."
        ],
        'contexts': [
            ['Paris is the capital and most populous city of France.', 'France is a country in Western Europe.'], 
            ['Harper Lee was an American novelist best known for her 1960 novel To Kill a Mockingbird.', 'The book is widely taught in schools in the United States.']
        ],
        'ground_truth': [
            "The capital of France is Paris.", 
            "Harper Lee wrote 'To Kill a Mockingbird'."
        ]
    }
    dataset = Dataset.from_dict(data_samples)

    # 2. 运行评估
    #    RAGAS会为数据集中的每一行，调用LLM作为裁判来计算各项指标的分数。
    print("Starting RAGAS evaluation...")
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,       # 答案是否忠于上下文
            answer_relevancy,   # 答案是否与问题相关
            context_precision,  # 检索到的上下文的相关性比例
            context_recall,     # 检索到的上下文覆盖了多少应有的信息
        ],
        # is_async=True # 如果数据集很大，可以开启异步模式加速
    )
    print("Evaluation complete.")

    # 3. 查看并解释结果
    print("\n--- RAGAS Evaluation Results ---")
    # result是一个字典，包含了每个指标的平均分
    print(result)

    # 将结果转换为更易读的DataFrame
    df = result.to_pandas()
    print("\n--- Detailed Results (DataFrame) ---")
    print(df.head())
    # 输出的DataFrame会展示每一条数据的各项指标得分，便于分析具体案例。
```

## **系统性能指标 (System Performance Metrics)**

- **端到端延迟 (End-to-End Latency):** 从用户提问到返回答案的总时间。
- **吞吐量 (Throughput):** 系统每秒能处理的请求数。
- **资源利用率:**
- **错误率和可用性:**

[**G1：综合评估指标** v2](https://www.notion.so/G1-v2-26055a58d45c80fdb253d3f203a294f6?pvs=21)