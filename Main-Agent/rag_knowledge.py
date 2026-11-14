# 只在新增 / 修改攻略的时候使用此代码
# 导入操作系统模块，用于文件夹遍历、路径判断、文件读取
import os
# 导入chromadb向量数据库库，用来存储攻略的向量
import chromadb
# dotenv：读取.env里面的环境变量（百炼API密钥等）
from dotenv import load_dotenv
# 文本分割器：把长攻略文本切分成小块（chunk），适合向量化检索
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 从helper2脚本导入get_embedding函数：调用百炼embedding接口，文本转向量
from helper2 import get_embedding

# 加载同目录下.env文件中的环境变量，会读取BAILIAN_API_KEY、BAILIAN_BASE_URL等配置
load_dotenv()

# 初始化Chroma持久化客户端，数据保存在当前目录下 chroma_db 文件夹
# 重点：和用户记忆共用同一个chroma数据库文件夹，但是使用【不同collection集合】，数据隔离
chroma_client = chromadb.PersistentClient(path="./chroma_db")
# 获取/创建名为 travel_knowledge_base 的集合，专门存放RAG攻略向量（和记忆集合agent_memory_index分开）
knowledge_collection = chroma_client.get_or_create_collection(name="travel_knowledge_base")

# 实例化递归字符文本切分器
# chunk_size：每个文本块最大字符长度
# chunk_overlap：相邻两个文本块重叠的字符，防止一句话被切断丢失上下文
# separators：分割优先级，优先按换行、句号、逗号切割，保证句子完整性
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80,
    separators=["\n\n", "\n", "。", "，"]
)

def load_folder_to_knowledge(folder_path: str = "./knowledge"):
    """
    加载knowledge文件夹下全部txt攻略入库RAG向量库
    :param folder_path: 存放攻略txt的文件夹路径，默认是当前目录下knowledge
    """
    # 判断知识库文件夹是否存在，如果不存在就新建这个文件夹
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"✅ 已创建知识库目录：{folder_path}，请放入txt攻略")
        return

    # 定义三个列表，分别存放：文本块、向量id、元数据（来源文件名）
    all_chunks = []
    all_ids = []
    all_metas = []
    # 自增索引，用来生成每个向量唯一id
    idx = 0

    # 遍历knowledge文件夹里面所有文件
    for fname in os.listdir(folder_path):
        # 只处理后缀为 .txt 的攻略文件，跳过其他格式
        if fname.endswith(".txt"):
            # 拼接得到txt文件的完整路径
            fpath = os.path.join(folder_path, fname)
            # 打开txt文件，utf-8编码读取里面的攻略文本
            with open(fpath, "r", encoding="utf-8") as f:
                raw = f.read()
            # 将读取到的长文本，切分成多个小块chunk
            chunks = text_splitter.split_text(raw)
            # 循环每一个切分好的文本块
            for chunk in chunks:
                all_chunks.append(chunk)
                # 给每个文本块生成唯一id，格式 know_0、know_1、know_2 ...
                all_ids.append(f"know_{idx}")
                # 元数据：记录这个chunk来自哪个txt文件，检索的时候可以展示来源
                all_metas.append({"source": fname})
                # 索引+1，给下一个chunk使用
                idx += 1

    # 如果切分出来的文本块数量为0，说明knowledge里面没有有效txt内容
    if len(all_chunks) == 0:
        print("⚠️ 知识库为空，请放入txt攻略文件到 ./knowledge")
        return

    # 【全量刷新逻辑】取出当前集合里全部已存在的向量id
    old_data = knowledge_collection.get()
    # 如果里面存在旧数据，就全部删除，避免新旧攻略混在一起
    if old_data["ids"]:
        knowledge_collection.delete(ids=old_data["ids"])

    # 批量循环，把每一段文本调用embedding接口转为向量
    embeddings = [get_embedding(c) for c in all_chunks]
    # 批量把文本、向量、id、元数据写入Chroma的travel_knowledge_base集合
    knowledge_collection.add(
        ids=all_ids,
        documents=all_chunks,
        embeddings=embeddings,
        metadatas=all_metas
    )
    # 打印完成提示，输出一共存入多少文本块
    print(f"✅ RAG知识库加载完成，共 {len(all_chunks)} 个文本块")



# 程序入口：直接运行这个py文件的时候，执行知识库加载重建逻辑
# 如果是被别的文件import导入，这一段不会自动执行
if __name__ == "__main__":
    load_folder_to_knowledge()
