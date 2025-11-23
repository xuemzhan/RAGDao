# E2：针对特定任务微调 (Task-Specific Fine-tuning)——教会模型“RAG式思考”

领域适应让模型“懂行”，而任务微调则让模型“会做事”。这里的“事”就是RAG的核心任务：**根据给定的上下文和问题，生成一个高质量的回答**。这要求模型不仅能理解信息，还能有效地整合、提炼，并遵循特定的指令。

## **开发模拟RAG过程的自定义数据集**

- **详细阐述：** 微调的关键是构建一个模拟RAG真实工作流的数据集。每一条训练样本都应该包含：用户查询 (Query)、检索到的上下文 (Context)、以及一个理想的回答 (Answer)。
- **数据生成策略：**
    - **人工标注：** 质量最高，成本也最高。
    - **合成数据 (Synthetic Data)：** 利用强大的LLM（如GPT-4）自身来生成数据。
        1. 从知识库中随机抽取文档块作为Context。
        2. 让LLM根据Context生成一个 plausible 的 Query。
        3. 再让LLM根据Context和Query生成一个理想的Answer。
- **[图片建议]:** 一张流程图展示合成数据的生成过程：Document Chunk -> LLM -> Question -> LLM -> Answer，最终形成一个(Question, Context, Answer)的三元组。

## **实施指令微调 (Instruction Fine-tuning)**

- **详细阐述：** 这种微调旨在提高模型遵循特定指令的能力。通过提供大量“指令-输入-期望输出”的范例，模型学会了理解指令的意图，并以期望的格式进行响应。
- **关键技术——PEFT与LoRA：**
    - **概念出处：** LoRA (Low-Rank Adaptation of Large Language Models) 由微软的研究人员在2021年的论文中提出，是PEFT（Parameter-Efficient Fine-Tuning）技术中最具代表性的一种。
    - **类比：** 想象一下**改装一辆汽车**来参加特定比赛。
        - **全模型微调：** 相当于把整台**引擎**都拆开重组。性能提升可能最大，但成本极高，且有把车改坏的风险（灾难性遗忘）。
        - **LoRA微调：** 相当于只给引擎加装一个**小型的涡轮增压器（Adapter）**。你只改动和训练这个小部件，而引擎主体保持不变。这能以很小的成本，获得接近重组引擎的性能提升，而且随时可以把这个部件拆下来，车子就恢复了原状。
- **可执行代码示例 (使用transformers, peft和datasets进行LoRA微调):**

```python
# 准备环境:
# pip install transformers peft datasets accelerate bitsandbytes torch

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling

# 1. 加载预训练模型和分词器
#    这里我们使用一个较小的模型 'EleutherAI/gpt-neo-125M' 以便快速演示。
#    在真实项目中，你会使用像Llama, Mistral等更强大的模型。
model_name = "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 使用4-bit量化加载模型，以在消费级GPU上运行
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    torch_dtype=torch.bfloat16
)

# 设置padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# 2. 准备模型进行PEFT训练
model = prepare_model_for_kbit_training(model)

# 3. 配置LoRA (参数高效微调)
lora_config = LoraConfig(
    r=8, # LoRA的秩，影响可训练参数量
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"], # 通常微调query和value投影层
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
print("Trainable parameters overview:")
model.print_trainable_parameters() 

# 4. 创建一个模拟的RAG指令微调数据集
data = {
    "text": [
        f"""### Instruction:
        Use the provided context to answer the question.
				Context:
				Document 1: The first RAG paper was published by Lewis et al. in 2020.
				Question:
				When was the first RAG paper published?
				Answer:
				The first RAG paper was published in 2020.""",
				f"""### Instruction:
				Use the provided context to answer the question.
				Context:
				Document 1: Paris is the capital and most populous city of France.
				Question:
				What is the capital of France?
				Answer:
				Paris is the capital of France."""]
}
dataset = Dataset.from_dict(data)

# 5. 对数据集进行Tokenization
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 6. 配置训练参数并启动训练
training_args = TrainingArguments(
    output_dir="./rag_finetuned_lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=1,
    fp16=True, # 使用16位浮点数进行训练
)

# DataCollatorForLanguageModeling 会自动处理padding和创建label
# label就是输入的token ID，模型的目标是预测下一个token
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print("\nStarting LoRA fine-tuning...")
trainer.train()
print("Fine-tuning complete.")

# 7. (可选) 推理测试
prompt = """### Instruction:
Use the provided context to answer the question.
Context:
Document 1: The headquarters of Google is located in Mountain View, California.
Question:
Where is Google's headquarters?
Answer:
"""
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
# 使用微调后的模型生成答案
outputs = model.generate(**inputs, max_new_tokens=20)
print("\n--- Inference Test ---")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

[**E2：针对特定任务微调 v2**](https://www.notion.so/E2-v2-26055a58d45c80e19a49f97f90206d24?pvs=21)