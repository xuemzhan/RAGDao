# C3：多文档综合策略 (Multi-Document Synthesis Strategies)

RAG系统通常会检索到多个相关的文档块。这些块可能内容互补、部分重叠，甚至相互矛盾。提示需要指导模型如何成为一个优秀的“信息整合者”和“冲突调解员”。

## **常用综合方法**

- **信息摘要 (Summarization):** 指示模型先总结每个文档的关键点，再整合成一个连贯的答案。
- **观点对比 (Viewpoint Comparison):** 对于包含不同观点或数据的文档，指示模型识别并呈现这些差异。
- **证据链构建 (Evidence Chaining):** 对于需要多步推理的问题，指示模型将多个文档的信息串联成一个逻辑清晰的证据链条。
- **置信度评估 (Confidence Scoring):** 指示模型根据多个来源信息的一致性，来评估最终答案的置信度。

## **制定多源综合信息的策略**

- **示例指令：** "Read all the provided documents carefully. Synthesize the information from all relevant documents to construct a comprehensive and coherent answer. Do not simply copy-paste sentences from the documents. Your answer should integrate the key points into a smooth narrative."

## **实施解决冲突信息的技术**

- **巧妙设计——“报告冲突”指令：**
    - **方法：** 指示模型在发现冲突时不应自行选择一方，而应将冲突本身作为答案的一部分呈现给用户。
    - **示例指令：** "If you find conflicting information across different documents, do not try to resolve it. Instead, you must present both viewpoints and explicitly state that there is a contradiction in the provided sources. For example: 'Document [1] states that the project deadline is on Monday, however, Document [3] states that it is on Wednesday.'"
    - **优点：** 这种方法极大地增强了系统的透明度和可信度，将最终的判断权交给了用户。

## **高级模式：Map-Reduce与Refine**

对于需要处理大量文档（例如，超过LLM上下文窗口长度）或进行深度摘要的场景，可以采用更复杂的提示链策略。

- **Map-Reduce 策略：**
    - **流程：**
        1. **Map 阶段:** 将所有检索到的文档块，**并行地**、**逐个地**喂给LLM，让它对**每一个**文档块都进行一个独立的初步处理（例如，生成一个摘要或提取关键信息）。
        2. **Reduce 阶段:** 将所有从Map阶段得到的初步结果（例如，所有的摘要），合并在一起，再喂给LLM一次，让它对这些**摘要的摘要**进行最终的综合和提炼，得出最终答案。
    - **类比：** **公司年度报告**的撰写。每个部门（Map）先各自提交一份年度总结，然后总经办（Reduce）将所有部门的总结汇总，提炼成一份面向董事会的最终报告。
- **Refine 策略：**
    - **流程：**
        1. 先将第一个文档块喂给LLM，生成一个初步答案。
        2. 然后，将第二个文档块和**上一步的答案**一起喂给LLM，让它根据新信息**修正和完善**已有的答案。
        3. 依次迭代，直到处理完所有文档。
    - **优点：** 能够构建更连贯、更有逻辑深度的答案。
    - **缺点：** 串行处理，耗时较长。