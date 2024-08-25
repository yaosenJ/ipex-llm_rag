import os
import torch
from typing import Any, List, Optional
from ipex_llm.transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, TextIteratorStreamer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core import QueryBundle, SimpleDirectoryReader
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores import VectorStoreQuery
import chromadb
import streamlit as st
from threading import Thread
import pandas as pd

# 设置OpenMP线程数为8
os.environ["OMP_NUM_THREADS"] = "2"

class Config:
    """配置类,存储所有需要的参数"""
    model_path = "./qwen2chat1.5_int4"
    tokenizer_path = "./qwen2chat1.5_int4"
    data_path = "./datas"
    persist_dir = "./chroma_db"
    embedding_model_path = "./qwen2chat_src/AI-ModelScope/bge-small-zh-v1___5"
    max_new_tokens = 1500

    def __init__(self):
        self.model = AutoModelForCausalLM.load_low_bit(self.model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True)
        self.embed_model = HuggingFaceEmbedding(model_name=self.embedding_model_path)

def load_vector_database(persist_dir: str) -> ChromaVectorStore:
    
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
   

    def __init__(self, vector_store: ChromaVectorStore, embed_model: Any, query_mode: str = "default", similarity_top_k: int = 2) -> None:
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_embedding = self._embed_model.get_query_embedding(query_bundle.query_str)
        vector_store_query = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=self._similarity_top_k, mode=self._query_mode)
        query_result = self._vector_store.query(vector_store_query)
        
        nodes_with_scores = [
            NodeWithScore(node=node, score=query_result.similarities[index] if query_result.similarities else None)
            for index, node in enumerate(query_result.nodes)
        ]
        print(f"Retrieved {len(nodes_with_scores)} nodes with scores")
        return nodes_with_scores

def generate_response(real_prompt, config: Config):
    tokenizer = config.tokenizer
    model = config.model
    
    text = tokenizer.apply_chat_template([{"role": "user", "content": real_prompt}], tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt")

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(inputs=model_inputs.input_ids, max_new_tokens=config.max_new_tokens, streamer=streamer)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    return streamer

def chat(config: Config, chat_destination, chat_departure, chat_days, chat_style, chat_budget, chat_people, chat_other):
    

    prompt_template = """你现在是一位专业的旅行规划师，你的责任是根据旅行出发地、目的地、天数、行程风格（紧凑、适中、休闲）、预算、随行人数，帮助我规划旅游行程并生成详细的旅行计划表。请你以表格的方式呈现结果。旅行计划表的表头请包含**日期**、**地点**、**行程计划**、**交通方式**、**餐饮安排**、**住宿安排**、**费用估算**、**备注**。

    下面是一个示例：
    ```
    | 日期 | 地点 | 行程计划 | 交通方式 | 餐饮安排 | 住宿安排 | 费用估算 | 备注 |
    |------|------|----------|----------|----------|----------|----------|------|
    | Day1 | 城市A | 上午：参观博物馆, 下午：游览公园 | 步行, 地铁 | 本地特色餐厅 | 酒店 | ￥500 | 天气晴 |
    | Day2 | 城市B | 上午：购物, 下午：休息 | 出租车 | 快餐 | 民宿 | ￥400 | 注意防晒 |
    ```
    请严格按照表格格式生成，表格内容根据以下旅行信息填写：
    旅游出发地：{}，旅游目的地：{} ，天数：{}天 ，行程风格：{} ，预算：{}，随行人数：{}, 特殊偏好、要求：{}

    """
    final_query = prompt_template.format(chat_departure, chat_destination, chat_days, chat_style, chat_budget, chat_people, chat_other)

    # 收集响应的所有部分并逐步输出到文本框
    response = ""
    placeholder = st.empty()  # 创建一个占位符

    streamer = generate_response(final_query, config)

    for i, chunk_text in enumerate(streamer):
        response += chunk_text
        # 逐步更新文本框内容，提供唯一的 key
        with placeholder.container():
            st.text_area("旅行计划表", value=response, height=400, key=f"response_text_{i}")

    return response

def parse_plan_to_table(plan_text: str) -> pd.DataFrame:
    # 解析文本到 DataFrame 表格
    lines = plan_text.strip().split('\n')
    table_data = [line.split('|')[1:-1] for line in lines[2:]]  # 忽略标题和边框行
    df = pd.DataFrame(table_data, columns=["日期", "地点", "行程计划", "交通方式", "餐饮安排", "住宿安排", "费用估算", "备注"])
    return df

def main():
    config = Config()
    # 初始化 st.session_state 中的 messages 列表
        
    is_arg = str()
    st.title("Wanderlust_Companion-旅游助手")

    # 通过侧边栏选择功能页面
    st.sidebar.title("ipex-llm推理部署旅游领域应用")
    assistant_type = st.sidebar.radio(
        "您的专属旅游助手",
        ("旅游规划助手", "旅游陪伴助手")
    )
    st.sidebar.image("logo.png")
    
    # 判断选择的助手类型
    if assistant_type == "旅游陪伴助手":
        is_arg = st.radio("是否使用RAG生成", ("Yes", "No"))

    # 页面1: 生成旅行计划表
    if assistant_type == "旅游规划助手":
        st.header("旅游规划助手")

        chat_departure = st.text_input("出发地", value="合肥")
        chat_destination = st.text_input("目的地", value="上海")
        chat_days = st.number_input("天数", min_value=1, max_value=30, value=3)
        chat_style = st.selectbox("行程风格", ["紧凑", "适中", "休闲"])
        chat_budget = st.number_input("预算 (元)", min_value=500, max_value=100000, value=3000)
        chat_people = st.number_input("随行人数", min_value=1, max_value=10, value=1)
        chat_other = st.text_input("特殊要求", value="无")

        if st.button("生成旅行计划表"):
            response = chat(config, chat_destination, chat_departure, chat_days, chat_style, chat_budget, chat_people, chat_other)
            if response:
                # 使用 DataFrame 显示表格
                df = parse_plan_to_table(response)
                st.table(df)

    # 页面2: 旅游陪伴助手
    elif assistant_type == "旅游陪伴助手":
        st.header("旅游陪伴助手")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
     # 显示聊天历史
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
        if prompt := st.chat_input("你可以询问关于旅游攻略问题"):
            if is_arg == "Yes":   
                vector_store = load_vector_database(persist_dir=config.persist_dir)
                query_embedding = config.embed_model.get_query_embedding(prompt)
                query_mode = "default"
                vector_store_query = VectorStoreQuery(
                    query_embedding=query_embedding, similarity_top_k=1, mode=query_mode
                )
                query_result = vector_store.query(vector_store_query)

                texts = " ".join([node.text for node in query_result.nodes])
                print(texts)
                final_prompt = f"你是一个旅游小助手，你的任务是，根据收集到的信息：\n{texts}.\n来回答用户所提出的问题：{prompt}。"
                real_prompt = final_prompt
            else:
                real_prompt = prompt

            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            response = ""
            with st.chat_message("assistant"):
                message_placeholder = st.empty()

                streamer = generate_response(real_prompt, config)
                for text in streamer:
                    response += text
                    message_placeholder.markdown(response + "▌")

                message_placeholder.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response})   

    # 清理消息历史
    with st.sidebar:
        clear_button = st.button("清除聊天历史")
        if clear_button:
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()