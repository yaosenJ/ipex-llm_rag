# 设置OpenMP线程数为8
import os
import time
os.environ["OMP_NUM_THREADS"] = "8"

import torch
from typing import Any, List, Optional
from ipex_llm.transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import TextStreamer, TextIteratorStreamer
# 从llama_index库导入HuggingFaceEmbedding类，用于将文本转换为向量表示
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# 从llama_index库导入ChromaVectorStore类，用于高效存储和检索向量数据
from llama_index.vector_stores.chroma import ChromaVectorStore
# 从llama_index库导入NodeWithScore和TextNode类
# NodeWithScore: 表示带有相关性分数的节点，用于排序检索结果
# TextNode: 表示文本块，是索引和检索的基本单位。节点存储文本内容及其元数据，便于构建知识图谱和语义搜索
from llama_index.core.schema import NodeWithScore, TextNode
# 从llama_index库导入QueryBundle类，用于封装查询相关的信息，如查询文本、过滤器等
from llama_index.core import QueryBundle
# 从llama_index库导入BaseRetriever类，这是所有检索器的基类，定义了检索接口
from llama_index.core.retrievers import BaseRetriever
# 从llama_index库导入SentenceSplitter类，用于将长文本分割成句子或语义完整的文本块，便于索引和检索
from llama_index.core.node_parser import SentenceSplitter
# 从llama_index库导入VectorStoreQuery类，用于构造向量存储的查询，支持语义相似度搜索
from llama_index.core.vector_stores import VectorStoreQuery
# 向量数据库
import chromadb
from ipex_llm.llamaindex.llms import IpexLLM
from llama_index.core import SimpleDirectoryReader
import streamlit as st
from threading import Thread

class Config:
    """配置类,存储所有需要的参数"""
    model_path = "./qwen2chat_int4"
    tokenizer_path = "./qwen2chat_int4"
    data_path = "./datas"
    persist_dir = "./chroma_db"
    embedding_model_path = "./qwen2chat_src/AI-ModelScope/bge-small-zh-v1___5"
    max_new_tokens = 512

def load_vector_database(persist_dir: str) -> ChromaVectorStore:
    """
    加载或创建向量数据库
    
    Args:
        persist_dir (str): 持久化目录路径
    
    Returns:
        ChromaVectorStore: 向量存储对象
    """
    # 检查持久化目录是否存在
    if os.path.exists(persist_dir):
        print(f"正在加载现有的向量数据库: {persist_dir}")
        chroma_client = chromadb.PersistentClient(path=persist_dir)
        chroma_collection = chroma_client.get_collection("travel")
    else:
        print(f"创建新的向量数据库: {persist_dir}")
        chroma_client = chromadb.PersistentClient(path=persist_dir)
        chroma_collection = chroma_client.create_collection("travel")
    print(f"Vector store loaded with {chroma_collection.count()} documents")
    return ChromaVectorStore(chroma_collection=chroma_collection)

def load_data(data_path: str) -> List[TextNode]:
    """
    加载并处理PDF数据
    
    Args:
        data_path (str): PDF文件路径
    
    Returns:
        List[TextNode]: 处理后的文本节点列表
    """
    # loader = PyMuPDFReader()
    # documents = loader.load(file_path=data_path)
    reader = SimpleDirectoryReader(
        input_dir=data_path,
        recursive=True,
        required_exts=[
            ".txt",
        ],
    )
    documents = reader.load_data()
    text_parser = SentenceSplitter(chunk_size=1024)
    text_chunks = []
    doc_idxs = []
    for doc_idx, doc in enumerate(documents):
        cur_text_chunks = text_parser.split_text(doc.text)
        text_chunks.extend(cur_text_chunks)
        doc_idxs.extend([doc_idx] * len(cur_text_chunks))

    nodes = []
    for idx, text_chunk in enumerate(text_chunks):
        node = TextNode(text=text_chunk)
        src_doc = documents[doc_idxs[idx]]
        node.metadata = src_doc.metadata
        nodes.append(node)
    return nodes

class VectorDBRetriever(BaseRetriever):
    """向量数据库检索器"""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embed_model: Any,
        query_mode: str = "default",
        similarity_top_k: int = 2,
    ) -> None:
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        检索相关文档
        
        Args:
            query_bundle (QueryBundle): 查询包
        
        Returns:
            List[NodeWithScore]: 检索到的文档节点及其相关性得分
        """
        query_embedding = self._embed_model.get_query_embedding(
            query_bundle.query_str
        )
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
        )
        query_result = self._vector_store.query(vector_store_query)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))
        print(f"Retrieved {len(nodes_with_scores)} nodes with scores")
        return nodes_with_scores



def main():
    """主函数"""
    config = Config()
    
    # 设置嵌入模型
    embed_model = HuggingFaceEmbedding(model_name=config.embedding_model_path)
    
    # 设置语言模型
    # 加载低位(int4)量化模型,trust_remote_code=True允许执行模型仓库中的自定义代码
    model = AutoModelForCausalLM.load_low_bit(config.model_path, trust_remote_code=True)
    # 加载对应的分词器
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path, trust_remote_code=True)
    # 创建文本流式输出器
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
      
     # 定义生成响应函数
    def generate_response(messages, message_placeholder):
        # 将用户的提示转换为消息格式
        # messages = [{"role": "user", "content": prompt}]
        # 应用聊天模板并进行 token 化
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt")

        # 创建 TextStreamer 对象，跳过提示和特殊标记
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        # 使用 zip 函数同时遍历 model_inputs.input_ids 和 generated_ids
        generation_kwargs = dict(inputs=model_inputs.input_ids, max_new_tokens=512, streamer=streamer)
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        return streamer

    # Streamlit 应用部分
    # 设置应用标题
    st.title("旅游助手")
    
    with st.sidebar:
        is_arg = st.radio(
            "Whether use RAG for generate",
            ("Yes", "No")
        )
        st.image("logo.png")
      
    clear_button = st.sidebar.button("清除聊天历史")

        # 当用户点击清除按钮时，清空聊天历史
    if clear_button:
        st.session_state.messages = []
        st.rerun()  # 重新运行应用以清除界面显示

    # 初始化聊天历史，如果不存在则创建一个空列表
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 显示聊天历史
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 用户输入部分
    if prompt := st.chat_input("你可以询问关于旅游攻略问题"):
        # 加载向量数据库
        vector_store = load_vector_database(persist_dir=config.persist_dir)
        
        # # 加载和处理数据
        # nodes = load_data(data_path=config.data_path)
        # for node in nodes:
        #     node_embedding = embed_model.get_text_embedding(
        #         node.get_content(metadata_mode="all")
        #     )
        #     node.embedding = node_embedding

        # # 将 node 添加到向量存储
        # vector_store.add(nodes)
        # 设置查询
        query_embedding = embed_model.get_query_embedding(prompt)

        # 执行向量存储检索
        print("开始执行向量存储检索")
        query_mode = "default"
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding, similarity_top_k=2, mode=query_mode
        )
        query_result = vector_store.query(vector_store_query)
        print(query_result)

        # 处理查询结果
        print("开始处理检索结果")
        texts = " "
        for index, text in enumerate(query_result.nodes):
            texts = texts + text.text
        # print(texts)    
        final_prompt = f"你是一个旅游攻略小助手，你的任务是，根据收集到的信息：\n{texts}.\n来精准回答用户所提出的问题：{prompt}。"
        print(final_prompt)
        if is_arg=="Yes":    
            real_prompt = final_prompt
        else:
            real_prompt = prompt
        # 将用户消息添加到聊天历史
        st.session_state.messages.append({"role": "user", "content": real_prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        response  = str()
        # 创建空的占位符用于显示生成的响应
        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            # 调用模型生成响应
            streamer = generate_response(st.session_state.messages, message_placeholder)
            for text in streamer:
                response += text
                message_placeholder.markdown(response + "▌")

            message_placeholder.markdown(response)

        # 将助手的响应添加到聊天历史
        st.session_state.messages.append({"role": "assistant", "content": response})   
       
    
if __name__ == "__main__":
    main()
