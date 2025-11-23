# G3：反馈回路 (Feedback Loop)——让用户成为你的“老师”

最真实的评估，来自于最终用户。建立一个有效的反馈回路，意味着为用户提供方便的渠道来表达他们对系统输出的看法，并将这些宝贵的信号系统性地收集、分析，并用于指导系统的持续改进。

- **反馈机制的类型：**
    - **显式反馈 (Explicit Feedback):** “顶/踩”按钮、五星评分。信号明确，质量高。
    - **隐式反馈 (Implicit Feedback):** 用户是否**复制**了答案内容？（积极信号）；用户是否**追问**了问题？（消极信号）。数据量大，但有噪音。
- **巧妙设计——“反馈驱动的微调数据” (Feedback-driven Fine-tuning Data):**
    - **类比：** 这就像一个**学生**交上作业后，**老师**（用户）批改了作业（**反馈**），指出了错误的地方。学生（系统）拿回作业本，不仅知道了对错，还把所有错题整理到了一个**错题本**里（**构建偏好数据集**）。下次考试前，他会专门复习这个错题本（**进行DPO微调**），确保不再犯同样的错误。
    - **核心流程：**
    
    ```mermaid
    flowchart TD
        A[用户使用RAG系统] --> B{用户给出'踩'反馈};
        B --> C["收集'坏'的问答对<br>(Query, Context, Bad_Answer)"];
        C --> D{人工或LLM修正};
        D --> E["生成'好'的答案<br>(Good_Answer)"];
        E --> F[构建偏好数据集<br>chosen: Good_Answer<br>rejected: Bad_Answer];
        F --> G{使用DPO等技术微调LLM};
        G --> H[部署微调后的新模型];
        H --> A;
    ```
    
    > DPO（Direct Preference Optimization）由斯坦福大学的研究人员在2023年的论文 "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" 中提出，是一种比传统的基于强化学习的RLHF更简单、更稳定的模型对齐方法。
    > 

[**G3：反馈回路** v2](https://www.notion.so/G3-v2-26055a58d45c8037a335c6f21c75d43e?pvs=21)