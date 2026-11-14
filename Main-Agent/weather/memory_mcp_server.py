from fastmcp import FastMCP
from dotenv import load_dotenv
import os
import sys
import pymysql
import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="travel_agent_memory_index")

mcp = FastMCP("agent-memory")

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

def get_mysql_conn():
    conn = pymysql.connect(
        host=os.getenv("MYSQL_HOST"),
        port=int(os.getenv("MYSQL_PORT")),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DB"),
        charset="utf8mb4"
    )
    return conn

def recall_memory(session_id: str, query: str, top_k=3):
    query_emb = get_embedding(query)
    result = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        where={"session_id": session_id}
    )
    if not result["metadatas"] or len(result["metadatas"][0]) == 0:
        return []

    mysql_id_list = [meta["mysql_id"] for meta in result["metadatas"][0]]
    conn = get_mysql_conn()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    fmt = ",".join(["%s"] * len(mysql_id_list))
    sql = f"SELECT id, session_id, memory_type, content FROM travel_agent_memory WHERE id IN ({fmt}) AND is_valid=1"
    cur.execute(sql, mysql_id_list)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

@mcp.tool()
async def recall_user_memory(session_id: str, query: str) -> str:
    """
    检索用户历史长期记忆，用来获取用户之前的出行信息（预算、人数、目的地、偏好等）
    Args:
        session_id: 会话ID，例如 travel_001
        query: 检索关键词，如：出行预算、出行人数、目的地、游玩偏好
    """
    memory_list = recall_memory(session_id, query, top_k=5)
    if not memory_list:
        return "暂无该会话的历史记忆"
    out = "\n".join(str(item) for item in memory_list)
    return out

# 如果你后面想扩展，还可以在这里加 save_memory / get_snapshot 工具

if __name__ == "__main__":
    mcp.run(transport="stdio")
