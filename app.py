import os
import torch
from typing import Any, List, Optional
from ipex_llm.transformers import AutoModelForCausalLM, AutoModelForSpeechSeq2Seq
from transformers import AutoTokenizer, TextIteratorStreamer, TextStreamer, AutoProcessor, WhisperProcessor
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
from PIL import Image
import time
import uuid
from io import BytesIO
from st_audiorec import st_audiorec
import librosa
import time
from pydub import AudioSegment
# 设置OpenMP线程数为24
os.environ["OMP_NUM_THREADS"] = "24"

class Config:
    """配置类,存储所有需要的参数"""
    model_path = "./models/shiqiyio/Qwen2-7B-Instruct_int4"
    tokenizer_path = "./models/shiqiyio/Qwen2-7B-Instruct_int4"
    data_path = "./datas"
    persist_dir = "./chroma_db"
    embedding_model_path = "./models/AI-ModelScope/bge-small-zh-v1___5"
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


# 模型加载，根据模型类型选择正确的加载方式
def load_model(model_path, low_bit=True):
    if low_bit:
        model1 = AutoModelForCausalLM.load_low_bit(model_path, trust_remote_code=True).eval()
        tokenizer1 = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        model1 = AutoModelForCausalLM.from_pretrained(model_path,
                                                 trust_remote_code=True,
                                                 load_in_low_bit="sym_int8",
                                                 _attn_implementation="eager",
                                                 modules_to_not_convert=["vision_embed_tokens"])
    
        # Load processor
        tokenizer1 = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model1, tokenizer1


# 图像理解模型推理
def image_understand(model1, tokenizer1, image_path):
    image = Image.open(image_path)
    query = '描述这张图片'
    # Load processor
    # processor = tokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # here the message formatting refers to https://huggingface.co/microsoft/Phi-3-vision-128k-instruct#sample-inference-code
    messages = [
        {"role": "user", "content": "<|image_1|>\n{prompt}".format(prompt=query)},
    ]
    prompt = tokenizer1.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    
    # Generate predicted tokens
    with torch.inference_mode():
        inputs = tokenizer1(prompt, [image], return_tensors="pt")
        st = time.time()
        output = model1.generate(**inputs,
                                eos_token_id=tokenizer1.tokenizer.eos_token_id,
                                num_beams=1,
                                do_sample=False,
                                max_new_tokens=128,
                                temperature=0.0)
        end = time.time()
        print(f'Inference time: {end-st} s')
        output_str = tokenizer1.decode(output[0],
                                      skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)
       
        print(output_str)
    
    return output_str

# 文本生成函数
def stream_model(model1, tokenizer1, question):
    messages = [{"role": "user", "content": question}]
    text = tokenizer1.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer1([text], return_tensors="pt")
    
    with torch.inference_mode():
        generated_ids = model1.generate(
            model_inputs.input_ids,
            max_new_tokens=1024,
            streamer=TextStreamer(tokenizer1, skip_prompt=True, skip_special_tokens=True),
        )
        generated_text = tokenizer1.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text

# 生成文案主逻辑
def generate_text_from_image(image, style, image_model, image_tokenizer, text_model, text_tokenizer):
    # temp_image_path = save_and_get_temp_url(image)
    image_description = image_understand(image_model, image_tokenizer, image)
    
    question = f"根据图片描述：{image_description}, 用{style}风格生成一段文字。"
    generated_text = stream_model(text_model, text_tokenizer, question)
    return generated_text

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
    
def recognize_speech_from_microphone(wav_audio_data):
    # 使用 st_audiorec 录制音频
    
    wav_audio_data = st_audiorec()
    
    if wav_audio_data is not None:
        # 将 WAV 格式的音频数据转换为 MP3
        audio = AudioSegment.from_wav(BytesIO(wav_audio_data))
        mp3_output = "audio1.mp3"
        
        # 导出为 MP3 文件，并确保写入完成
        audio.export(mp3_output, format="mp3")
        
        st.success(f"录音已保存为 {mp3_output}")
        
        # 在 MP3 文件写入完成之后继续执行以下代码
        model2 = AutoModelForSpeechSeq2Seq.from_pretrained(pretrained_model_name_or_path="./models/AI-ModelScope/whisper-large-v3",
                                                          load_in_4bit=True,
                                                          trust_remote_code=True)
        processor = WhisperProcessor.from_pretrained(pretrained_model_name_or_path="./models/AI-ModelScope/whisper-large-v3",
                                                     trust_remote_code=True)
        
        # 加载音频数据并进行采样率转换
        data_en, sample_rate_en = librosa.load(mp3_output, sr=16000)
        
        # 定义任务类型
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="Chinese", task="transcribe")
        
        with torch.inference_mode():
            # 为 Whisper 模型提取输入特征
            input_features = processor(data_en, sampling_rate=sample_rate_en, return_tensors="pt").input_features
            
            # 为转录预测 token id
            st_time = time.time()
            predicted_ids = model2.generate(input_features)
            end_time = time.time()
            
            # 将 token id 解码为文本
            transcribe_str = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            
            print(f'Inference time: {end_time - st_time} s')
            print('-' * 20, 'Chinese Transcription', '-' * 20)
            print(transcribe_str)
            
            # 在 Streamlit 页面上显示转录文本
    return transcribe_str
    
def main():
    config = Config()
    # 初始化 st.session_state 中的 messages 列表
        
    is_arg = str()
    st.title("Wanderlust_Companion-旅游助手")

    # 通过侧边栏选择功能页面
    st.sidebar.title("ipex-llm推理部署旅游领域应用")
    assistant_type = st.sidebar.radio(
        "您的专属旅游助手",
        ("旅游规划助手", "旅游陪伴助手","旅游文案助手")
    )
    st.sidebar.image("logo.png")
    
    # 判断选择的助手类型
    if assistant_type == "旅游陪伴助手":
        st.header("旅游陪伴助手")
        is_arg = st.radio("选择问答模式", ("简单模式", "知识库模式","语音模式"))

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
        # st.header("旅游陪伴助手")
           
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
     # 显示聊天历史
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
        if prompt := st.chat_input("你可以询问关于旅游攻略问题"):
            
            if is_arg == "知识库模式":   
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
            if is_arg == "简单模式":
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
                
        
        if is_arg == "语音模式":
            wav_audio_data = st_audiorec()
           
    
            if wav_audio_data is not None:
                # 将 WAV 格式的音频数据转换为 MP3
                audio = AudioSegment.from_wav(BytesIO(wav_audio_data))
                mp3_output = "audio1.mp3"
                
                # 导出为 MP3 文件，并确保写入完成
                audio.export(mp3_output, format="mp3")
                
                # st.success(f"录音已保存为 {mp3_output}")
                
                # 在 MP3 文件写入完成之后继续执行以下代码
                model2 = AutoModelForSpeechSeq2Seq.from_pretrained(pretrained_model_name_or_path="./models/AI-ModelScope/whisper-large-v3",
                                                                  load_in_4bit=True,
                                                                  trust_remote_code=True)
                processor = WhisperProcessor.from_pretrained(pretrained_model_name_or_path="./models/AI-ModelScope/whisper-large-v3",
                                                             trust_remote_code=True)
                
                # 加载音频数据并进行采样率转换
                data_en, sample_rate_en = librosa.load(mp3_output, sr=16000)
                
                # 定义任务类型
                forced_decoder_ids = processor.get_decoder_prompt_ids(language="Chinese", task="transcribe")
                
                with torch.inference_mode():
                    # 为 Whisper 模型提取输入特征
                    input_features = processor(data_en, sampling_rate=sample_rate_en, return_tensors="pt").input_features
                    
                    # 为转录预测 token id
                    st_time = time.time()
                    predicted_ids = model2.generate(input_features)
                    end_time = time.time()
                    
                    # 将 token id 解码为文本
                    prompt = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                    
                    print(f'Inference time: {end_time - st_time} s')
                    print('-' * 20, 'Chinese Transcription', '-' * 20)
                    print(prompt)                    
                    #st.markdown(f"语音输入：{prompt}")
                    real_prompt = prompt
                time.sleep(15) 
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
                  
             
        
    elif assistant_type == "旅游文案助手":
        st.header("旅游文案助手")
        # 模型加载
        # image_model_path = "/dev/shm/Phi-3-vision-128k-instruct_int4"
        image_model_path = './models/LLM-Research/Phi-3-vision-128k-instruct'
        text_model_path = "./models/shiqiyio/Qwen2-7B-Instruct_int4"
        
    
        save_dir = "./uploaded_images/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 设置保存文件的特定名称
        filename = "uploaded_image.jpg"
        
        # 创建文件上传的UI
        uploaded_file = st.file_uploader("上传一张图片", type=["jpg", "jpeg", "png"])
        style_options = ["朋友圈", "小红书", "微博", "抖音"]
        # uploaded_file = st.file_uploader("上传图像", type=["png", "jpg", "jpeg"])
        style_dropdown = st.selectbox("选择风格模式", style_options)
        # generate_button = st.button("生成文案")
        
        if uploaded_file is not None:
            # 设置保存路径（使用特定的文件名）
            file_path = os.path.join(save_dir, filename)
            
            # 将文件保存到指定目录，复写文件
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            # 显示成功信息
            st.success(f"图片已加载，请稍后...")
            if os.path.exists(file_path):
                # 图片保存成功后，展示图片
                st.image(file_path)
                
                # 生成文案
                image_model, image_tokenizer = load_model(image_model_path, low_bit=False)  # 低位量化模型
                text_model, text_tokenizer = load_model(text_model_path, low_bit=True)      # 低位量化模型
                
                generated_text = generate_text_from_image(file_path, style_dropdown, image_model, image_tokenizer, text_model, text_tokenizer)
                st.text_area("生成的文案", value=generated_text, height=500)
                
                
            else:
                st.error("保存图像时出错。")
        else:
            st.warning("请上传一张图像。")
    # 清理消息历史
    with st.sidebar:
        clear_button = st.button("清除聊天历史")
        if clear_button:
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()
