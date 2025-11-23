# 开篇：针对RAG微调语言模型

> **当RAG遇上微调——从“通用向导”到“领域大师”**
> 

在RAG的核心哲学中，LLM通常被视为一个“开箱即用”的、知识渊博的“通用向导”。它的能力主要依赖于我们（通过检索器）为它提供的“旅行手册”（外部知识）。然而，这并非故事的全部。在追求极致性能、解决特定领域挑战，或实现高度可控的生成行为时，我们可能需要对这位向导进行“岗前特别培训”。这就是**微调（Fine-tuning）**。

微调，不是给向导灌输新的地图知识（那是RAG检索器的任务），而是教会他如何**更好地阅读地图、如何使用特定领域的行话与客户沟通、如何根据客户的要求定制旅行路线**。它让模型从“有礼貌的通用回答”，转变为“深度契合领域特点的权威回答”。

本章将作为您的“高级培训师指南”，探讨将微调融入RAG体系的四大核心策略，帮助您将您的LLM从一个“通用向导”，打造成一位真正的“领域大师”。

```mermaid
graph TD
    A["通用预训练LLM<br>(如 Llama 2, GPT-3.5)"] --> B{微调策略};

    subgraph "E1: 领域适应"
        C1["持续预训练<br>(在海量领域文本上)"]
        C1 --> D[目标: 让模型'懂行'<br>熟悉领域术语和语言风格]
    end

    subgraph "E2: 任务微调"
        C2["指令微调<br>(使用'Query-Context-Answer'三元组)"]
        C2 --> E[目标: 让模型'会做RAG'<br>学习如何利用上下文回答问题]
    end

    subgraph "E3: 检索感知训练"
        C3["RAFT等高级方法<br>(在含噪音的上下文中训练)"]
        C3 --> F[目标: 提升鲁棒性<br>学会在不完美检索结果中筛选信息]
    end
    
    subgraph "E4: 控制效率"
        C4[针对特定风格/格式微调]
        C4 --> G[目标: 输出可控<br>如强制JSON输出、摘要长度等]
    end

    B --> C1;
    B --> C2;
    B --> C3;
    B --> C4;

    D & E & F & G --> H[微调后的RAG专用LLM];
```

[**E1：领域适应 (Domain Adaptation)——让模型成为领域专家**](https://www.notion.so/E1-Domain-Adaptation-26055a58d45c80b79c0cef14b35c3c42?pvs=21)

[**E2：针对特定任务微调 (Task-Specific Fine-tuning)——教会模型“RAG式思考”**](https://www.notion.so/E2-Task-Specific-Fine-tuning-RAG-26055a58d45c809ab6eaced31503772e?pvs=21)

[**E3：检索感知训练 (Retrieval-Aware Training)——让生成器更懂检索器**](https://www.notion.so/E3-Retrieval-Aware-Training-26055a58d45c8002a357d448b8de1e9a?pvs=21)

[**E4：控制效率 (Control Efficiency)——微调生成风格与内容**](https://www.notion.so/E4-Control-Efficiency-26055a58d45c80a997bce18229af8659?pvs=21)