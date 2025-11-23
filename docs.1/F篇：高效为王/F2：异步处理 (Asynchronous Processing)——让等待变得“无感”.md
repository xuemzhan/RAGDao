# F2：异步处理 (Asynchronous Processing)——让等待变得“无感”

同步处理意味着用户发起一个请求后，必须一直等待直到服务器完成所有处理并返回结果。异步处理则允许服务器在收到请求后，立即返回一个“已受理”的响应，然后在后台处理任务。去一家**热门餐厅**吃饭。

- **同步（排队）：** 你必须站在门口一直排队，直到有空位，然后进去点餐、等待上菜。在整个过程中，你被“阻塞”了，不能做任何其他事。
- **异步（取号）：** 你在门口取一个号，然后就可以自由活动了，可以去逛逛街、喝杯咖啡。当轮到你的时候，餐厅会通过手机短信通知你回来就餐。你等待的时间变得“无感”了。
- **核心实现：**
    - **异步API端点:** 使用像**FastAPI (Python)**, Node.js等原生支持异步的Web框架。这极大地提高了服务器的并发处理能力（吞吐量）。
    - **后台任务队列:** 使用**Celery**, RabbitMQ, Kafka等消息队列系统。适用于处理耗时极长的复杂任务。
- **可执行代码示例 (使用FastAPI实现异步API):**

```python
# 准备环境:
# pip install fastapi uvicorn "python-multipart"
# 保存为 main.py, 然后在终端运行: uvicorn main:app --reload

from fastapi import FastAPI
import asyncio
import time

app = FastAPI()

# 模拟异步的RAG服务调用 (非阻塞)
async def execute_rag_pipeline(query: str):
    """模拟一个包含多个异步I/O操作的RAG流程"""
    print(f"[{time.strftime('%X')}] Starting RAG for query: '{query}'")
    
    # 1. 模拟异步调用检索器 (例如，一个异步的数据库客户端)
    # asyncio.sleep() 会让出CPU控制权，允许服务器处理其他请求。
    await asyncio.sleep(0.5) 
    retrieved_docs = ["doc1", "doc2"]
    print(f"[{time.strftime('%X')}] Docs retrieved for '{query}'")
    
    # 2. 模拟异步调用LLM API (例如，使用aiohttp库)
    await asyncio.sleep(1.5)
    answer = f"This is the generated answer for '{query}'."
    print(f"[{time.strftime('%X')}] Answer generated for '{query}'")
    
    return {"query": query, "answer": answer}

# API端点是异步的 (async def)
@app.post("/ask")
async def ask_question(query: str = "What is RAG?"):
    """这个API端点会异步执行RAG流程"""
    result = await execute_rag_pipeline(query)
    return result

# --- 如何测试 ---
# 1. 运行 `uvicorn main:app --reload`
# 2. 打开两个终端。
# 3. 在第一个终端快速运行: curl -X POST "http://127.0.0.1:8000/ask?query=Query_A"
# 4. 立即在第二个终端运行: curl -X POST "http://127.0.0.1:8000/ask?query=Query_B"
# 5. 观察运行uvicorn的终端日志。你会看到两个请求的处理过程是交织在一起的，
#    而不是一个完全结束后才开始下一个。这就是异步的威力。
```

[**F2：异步处理** v2](https://www.notion.so/F2-v2-26055a58d45c803d8f85d13f613581e6?pvs=21)