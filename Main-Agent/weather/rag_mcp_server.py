from fastmcp import FastMCP
from dotenv import load_dotenv
from openai import OpenAI
import os
import chromadb
import json

mcp = FastMCP("travel-knowledge-rag")

load_dotenv()

chroma_client = chromadb.PersistentClient(path="../Main-Agent/chroma_db")
knowledge_collection = chroma_client.get_or_create_collection(name="travel_knowledge_base")

def _get_bailian_client():
    client = OpenAI(
        api_key=os.getenv("BAILIAN_API_KEY"),
        base_url=os.getenv("BAILIAN_BASE_URL")
    )
    return client

def get_embedding(text: str):
    client = _get_bailian_client()
    resp = client.embeddings.create(
        input=text,
        model="text-embedding-v2"
    )
    return resp.data[0].embedding

def _search_travel_knowledge(query: str, top_k: int = 3) -> str:
    """
    检索旅游攻略知识库，返回拼接好的文本，供LLM使用
    Args:
        query: 检索关键词
        top_k: 返回几条结果

    """
    # 将用户查询的问句转为向量
    emb = get_embedding(query)
    # 在travel_knowledge_base集合中做向量相似度检索
    res = knowledge_collection.query(
        query_embeddings=[emb],
        n_results=top_k
    )
    # 安全判断：如果没有检索到任何结果，返回提示文字
    if not res["documents"] or len(res["documents"][0]) == 0:
        return "暂无相关攻略资料"

    chunks = res["documents"][0]
    output = ""
    for doc, meta in zip(chunks, res["metadatas"][0]):
        output += f"【来源：{meta['source']}】\n{doc}\n\n"

    return json.dumps({"content": output, "chunks": chunks}, ensure_ascii=False)

@mcp.tool()
async def search_travel_knowledge(query: str) -> str:
    """
    检索旅游攻略知识库，查询景点、美食、游玩贴士、当地特色
    Args:
        query: 查询内容，例如：长沙适合情侣的自然景点、长沙特色美食
    """
    return _search_travel_knowledge(query, top_k=3)

if __name__ == "__main__":
    mcp.run(transport="stdio")
