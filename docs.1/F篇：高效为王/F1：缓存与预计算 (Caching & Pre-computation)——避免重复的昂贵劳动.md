# F1：缓存与预计算 (Caching & Pre-computation)——避免重复的昂贵劳动

RAG管道中的许多步骤，如文档嵌入、LLM调用等，都是计算密集型且耗时的操作。缓存（Caching）的核心思想是“记住”之前计算过的结果。预计算（Pre-computation）则是将所有可以离线完成的计算，都提前处理好。这就像一位**高效的厨师**。

- **预计算：** 他会在每天开店前，就把所有的蔬菜洗好切好，高汤熬好（**预计算嵌入**）。
- **缓存：** 对于菜单上最热门的菜品（**热点问题**），他会提前预制几份半成品。当客人点单时，他可以直接加热装盘，而不是从头现做（**查询/LLM响应缓存**）。

## **缓存策略**

实施智能的缓存失效策略，确保缓存内容的时效性和准确性。

- **缓存失效策略**
    - **LRU (Least Recently Used):** 优先淘汰最久未被使用的缓存。
    - **TTL (Time-To-Live):** 基于时间的失效机制，为每个缓存条目设置一个有效期。
    - **事件驱动失效:** 当知识库中的源文档被修改时，主动触发一个事件来清除相关的缓存。
- **查询缓存：** 对完全相同的用户查询，直接返回之前缓存的最终答案。
- **LLM响应缓存：** 将完整的Prompt（包含指令、上下文、问题）作为Key，将LLM的生成结果作为Value进行缓存。这是**性价比最高**的缓存策略，能直接省去最昂贵的LLM调用步骤。
- **可执行代码示例 (使用functools.lru_cache实现简单的内存缓存):**

```python
# 准备环境:
# Python内置库，无需安装。

from functools import lru_cache
import time
import hashlib

# 模拟一个昂贵的LLM API调用
def call_llm_api(prompt: str) -> str:
    """模拟一个耗时2秒的LLM API调用"""
    print(f"\n--- [!] Calling Real LLM API (Expensive Operation) for prompt hash: {hashlib.md5(prompt.encode()).hexdigest()[:8]}... ---")
    time.sleep(2)
    return f"This is the generated answer for your query."

# 使用LRU (Least Recently Used) 缓存装饰器
# @lru_cache 会自动缓存函数的输入和输出。
# maxsize 定义了缓存中可以存储的最大条目数。当缓存满了之后，最久未被使用的条目会被丢弃。
@lru_cache(maxsize=128)
def cached_llm_call(prompt: str) -> str:
    """这是一个带缓存的LLM调用函数"""
    return call_llm_api(prompt)

# --- 模拟RAG流程 ---
def get_answer(query: str, context: str):
    # 实际的prompt会更复杂
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    
    print(f"Processing query: '{query}'")
    start_time = time.time()
    # 调用带缓存的函数
    response = cached_llm_call(prompt)
    duration = time.time() - start_time
    print(f"Response: '{response}'")
    print(f"Time taken: {duration:.4f} seconds")
    return response

# --- 第一次调用 ---
# 一个全新的查询和上下文组合
get_answer("What is RAG?", "RAG stands for Retrieval-Augmented Generation.")

# --- 第二次调用 (完全相同的查询和上下文) ---
# 这将触发缓存
get_answer("What is RAG?", "RAG stands for Retrieval-Augmented Generation.")

# --- 第三次调用 (上下文变了，即使查询相同，也会触发新的API调用) ---
get_answer("What is RAG?", "RAG is a technique to improve LLM accuracy.")

# 观察输出，你会发现只有第一次和第三次调用会真正执行 "call_llm_api"，
# 第二次调用会瞬间返回结果。
```

• **一个关键挑战——缓存失效 (Cache Invalidation):** 当知识库中的文档更新后，与该文档相关的缓存必须被清除，否则用户会得到过时的信息。设计一个健壮的缓存失效策略是缓存系统中最具挑战性的部分。

[**F1：缓存与预计算** v2](https://www.notion.so/F1-v2-26055a58d45c80319fd0fd870ab43553?pvs=21)