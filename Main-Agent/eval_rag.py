from ragas import evaluate, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings
import os
import json
from dotenv import load_dotenv
from ragas.run_config import RunConfig

load_dotenv()

# ========== 评估 LLM：DeepSeek ==========
eval_llm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
    temperature=0
)
evaluator_llm = LangchainLLMWrapper(eval_llm)

# ========== 【修改这里】DashScope官方Embedding，不再用OpenAIEmbeddings套壳 ==========
bailian_emb = DashScopeEmbeddings(
    model=os.getenv("BAILIAN_EMB_MODEL", "text-embedding-v2"),
    dashscope_api_key=os.getenv("BAILIAN_API_KEY")
)
evaluator_emb = LangchainEmbeddingsWrapper(bailian_emb)

# ========== 读取 Agent 生成的评测数据 ==========
with open("eval_data.json", "r", encoding="utf-8") as f:
    eval_data = json.load(f)

# 兜底清洗context
for row in eval_data:
    clean_ctx = []
    for ctx in row["retrieved_contexts"]:
        if isinstance(ctx, list):
            for sub_ctx in ctx:
                if isinstance(sub_ctx, str) and sub_ctx.strip():
                    clean_ctx.append(sub_ctx.strip())
        elif isinstance(ctx, str) and ctx.strip():
            clean_ctx.append(ctx.strip())
        elif isinstance(ctx, dict) and "text" in ctx and ctx["text"].strip():
            clean_ctx.append(ctx["text"].strip())
    row["retrieved_contexts"] = clean_ctx

dataset = EvaluationDataset.from_list(eval_data)

run_config = RunConfig(
    max_workers=1,
    max_retries=2
)

result = evaluate(
    dataset=dataset,
    metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision()],
    llm=evaluator_llm,
    embeddings=evaluator_emb,
    run_config=run_config
)

print(result)
df = result.to_pandas()
df.to_csv("rag_eval_result.csv", index=False, encoding="utf-8-sig")
print("✅ 结果已保存 rags_eval_result.csv")
