#  Wanderlust_Companion——智能旅游助手

<p align="center">
  <img src="https://github.com/yaosenJ/ipex-llm_rag/blob/main/logo.png" style="zoom:1%;">
</p>

## 1. 项目介绍

亲爱的旅游爱好者们，欢迎来到**Wanderlust_Companion——智能旅游助手** ，您的专属旅行伙伴！我们致力于为您提供个性化的旅行规划、陪伴和分享服务，让您的旅程充满乐趣并留下难忘回忆。

“Wanderlust_Companion——智能旅游助手”基于**英特尔软硬件环境**以及使用RAG、低精度量化、文生文、图生文以及语音识别等技术，旨在为您量身定制一份满意的旅行计划。无论您期望体验何种旅行目的地、天数、行程风格（如紧凑、适中或休闲）、预算以及随行人数，我们的助手都能为您精心规划行程并生成详尽的旅行计划表，包括每天的行程安排、交通方式以及需要注意的事项等。

此外，我们还采用RAG技术，专为提供实用全方位信息而设计，包括景点推荐、活动安排、餐饮、住宿、购物、行程推荐以及实用小贴士等。目前，我们的知识库已涵盖全国各地区、城市的旅游攻略信息，为您提供丰富多样的旅行建议。

您还可以随时拍摄旅途中的照片，并通过我们的应用上传。应用将自动为您生成适应不同社交媒体平台（如朋友圈、小红书、抖音、微博）的文案风格，让您轻松分享旅途中的点滴，与朋友们共同感受旅游的乐趣。

立即加入“Wanderlust_Companion”，让我们为您的旅行保驾护航，共同打造一段难忘的旅程！


 **功能模块**

- 原生**Qwen-2**大语言模型的旅游问答服务：提供基础的旅游咨询服务，回答用户关于旅游目的地、景点和活动的各种问题；
- 增强**Qwen-2**大语言模型旅游问答服务：通过从向量数据库**chroma**检索相关知识，提升模型对旅游问题的回答准确性和全面性；
- **Whisper**模型的语音识别：实现语音到文本的转换，允许用户通过语音输入进行旅游相关查询和互动；
- **Phi-3-vision**模型的旅游图片文案撰写：自动生成与旅游图片相关的详细文案，提供对景点的生动描述和推荐。

  
**技术亮点**

- 充分使用**ipex-llm工具**，高效优化模型以及在**阿里云ECS基于英特尔至强可扩展处理器ecs.g8i.6xlarge**部署旅游助手应用
- 旅游规划、文案生成**Prompt**高效设计
- 多模态生成：**图生文**，**ASR**
- 有效整合三大旅游助手:**旅游规划助手**、**旅游陪伴助手**、**旅游文案助手**
- RAG技术、高效检索知识库知识

## 2. 系统架构图

<center><img src="https://github.com/yaosenJ/ipex-llm_rag/blob/main/%E6%9E%B6%E6%9E%84%E5%9B%BE.png" alt="image-20240131182121394" style="zoom:100%;" />

## 3. 项目演示

- 旅游规划助手
<p align="center">
  <img src="https://github.com/yaosenJ/ipex-llm_rag/blob/main/%E6%97%85%E6%B8%B8%E8%A7%84%E5%88%92%E5%8A%A9%E6%89%8B.png">
</p>

- 旅游陪伴助手
<p align="center">
  <img src="https://github.com/yaosenJ/ipex-llm_rag/blob/main/%E6%97%85%E6%B8%B8%E9%99%AA%E4%BC%B4%E5%8A%A9%E6%89%8B.png" >
</p>

- 旅游文案助手
<p align="center">
  <img src="https://github.com/yaosenJ/ipex-llm_rag/blob/main/%E6%97%85%E6%B8%B8%E6%96%87%E6%A1%88%E5%8A%A9%E6%89%8B.png">
</p>
  
- 部署及演示视频
  
[Wanderlust_Companion——智能旅游助手: https://www.bilibili.com/video/BV1BcsMerEXx](https://www.bilibili.com/video/BV1BcsMerEXx)


**开源不易，如果本项目帮到大家，可以右上角帮我点个 star~ ⭐⭐ , 您的 star ⭐是我们最大的鼓励，谢谢各位！** 

## 4. 项目部署

### 4.1 平台选择

利用阿里云ECS基于英特尔至强可扩展处理器**ecs.g8i.6xlarge**付费实例作为复赛项目代码基准平台进行部署、调优及演示

**24核(vCPU) 96GiB** **ubuntu_22_04_x64_20G_alibase_20240710.vhd**  **100 Mbps(峰值)**

<p align="center">
    <br>
    <img src="https://github.com/yaosenJ/ipex-llm_rag/blob/main/ecs.g8i.6xlarge.png" />
    <br>
</p>

### 4.2 虚拟环境创建

```shell

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda --version
git clone https://github.com/yaosenJ/ipex-llm_rag.git
cd ipex-llm_rag
conda create -n ipex python=3.11 -y
conda activate ipex

```
### 4.3 环境依赖包安装

```shell
pip install -r requirements.txt
```
### 4.4 模型下载

运行下面代码块，即python down_model.py,下载 **Phi-3-vision-128k-instruct**, **Qwen2-7B-Instruct_int4**, **whisper-large-v3**, **bge-small-zh-v1.5**
```python

from modelscope import snapshot_download
model_dir = snapshot_download('LLM-Research/Phi-3-vision-128k-instruct',cache_dir ='./models/')
model_dir = snapshot_download('shiqiyio/Qwen2-7B-Instruct_int4',cache_dir ='./models/') #使用ipex-llm工具进行int4低精度量化
model_dir = snapshot_download('AI-ModelScope/whisper-large-v3',cache_dir ='./models/')
model_dir = snapshot_download('AI-ModelScope/bge-small-zh-v1.5',cache_dir ='./models/')

```

### 4.5 模型量化

运行下面代码块，即可对Qwen2-7B-Instruct进行int4量化
```python

from ipex_llm.transformers import AutoModelForCausalLM
from transformers import  AutoTokenizer
import os
if __name__ == '__main__':
    model_path = os.path.join(os.getcwd(),"qwen/Qwen2-7B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(model_path, load_in_low_bit='sym_int4', trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.save_low_bit('Qwen2-7B-Instruct_int4')
    tokenizer.save_pretrained('Qwen2-7B-Instruct_int4')

```

### 4.6 推理部署

```python
streamlit run app.py
```


### 5. 功能模块详解

- ASR
- TTS
- 图生文
- RAG
  
####  5.1 ASR

模型下载：[https://www.modelscope.cn/models/AI-ModelScope/whisper-large-v3](https://www.modelscope.cn/models/AI-ModelScope/whisper-large-v3)

```python
from modelscope import snapshot_download
model_dir = snapshot_download('AI-ModelScope/whisper-large-v3', cache_dir='./model/asr', revision='master' )
```

模型低精度**int4**量化
```python
from ipex_llm.transformers import AutoModelForSpeechSeq2Seq
from transformers import  AutoTokenizer

model =AutoModelForSpeechSeq2Seq.from_pretrained(pretrained_model_name_or_path="/mnt/workspace/A/AI-ModelScope/whisper-large-v3/",
                                                  load_in_4bit=True,
                                                  trust_remote_code=True)
model.save_low_bit('./model/asr/AI-ModelScope/whisper-large-v3_int4')
tokenizer.save_pretrained('./model/asr/AI-ModelScope/whisper-large-v3_int4')
```

加载量化版的**Whisper-large-v3**模型
```python
load_path = "./model/asr/AI-ModelScope/whisper-large-v3_int4"
model = AutoModelForSpeechSeq2Seq.load_low_bit(load_path, trust_remote_code=True)
```
加载 **Whisper Processor**

```python
from transformers import WhisperProcessor
processor = WhisperProcessor.from_pretrained(pretrained_model_name_or_path="./model/asr/A/AI-ModelScope/whisper-large-v3")
```
使用带有 INT4 优化功能的 **IPEX-LLM**优化 **Whisper-large-v3** 模型并加载 Whisper Processor 后，就可以开始通过模型推理转录音频了。
首先从原始语音波形中提取序列数据
```python
import librosa
data_en, sample_rate_en = librosa.load("audio_zh.mp3", sr=16000)
```
然后根据序列数据转录音频文件

```python
import torch
import time

# 定义任务类型
forced_decoder_ids = processor.get_decoder_prompt_ids(language="Chinese", task="transcribe")

with torch.inference_mode():
    # 为 Whisper 模型提取输入特征
    input_features = processor(data_en, sampling_rate=sample_rate_en, return_tensors="pt").input_features

    # 为转录预测 token id
    st = time.time()
    predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
    end = time.time()

    # 将 token id 解码为文本
    transcribe_str = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    print(f'Inference time: {end-st} s')
    print('-'*20, 'Chinese Transcription', '-'*20)
    print(transcribe_str)
```
最后结果展示如下：
<p align="center">
    <br>
    <img src="asr.png" />
    <br>
</p>

####5.2 图生文


```python

import os
import time
import torch
import argparse
import requests

from PIL import Image
from ipex_llm.transformers import AutoModelForCausalLM
from transformers import AutoProcessor

if __name__ == '__main__':
    model_path = "./models/LLM-Research/Phi-3-vision-128k-instruct"
    image_path = "./travel"

    query = '描述这张图片'
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 trust_remote_code=True,
                                                 load_in_low_bit="sym_int8",
                                                 _attn_implementation="eager",
                                                 modules_to_not_convert=["vision_embed_tokens"])
    
    # Load processor
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
   
    messages = [
        {"role": "user", "content": "<|image_1|>\n{prompt}".format(prompt=query)},
    ]
    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if os.path.exists(image_path):
       image = Image.open(image_path)
    else:
       image = Image.open(requests.get(image_path, stream=True).raw)
    
    # Generate predicted tokens
    with torch.inference_mode():
        inputs = processor(prompt, [image], return_tensors="pt")
        st = time.time()
        output = model.generate(**inputs,
                                eos_token_id=processor.tokenizer.eos_token_id,
                                num_beams=1,
                                do_sample=False,
                                max_new_tokens=128,
                                temperature=0.0)
        end = time.time()
        print(f'Inference time: {end-st} s')
        output_str = processor.decode(output[0],
                                      skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)
        print('-'*20, 'Prompt', '-'*20)
        print(f'Message: {messages}')
        print(f'Image link/path: {image_path}')
        print('-'*20, 'Output', '-'*20)
        print(output_str)

```

最后结果展示如下：
<p align="center">
    <br>
    <img src="asr.png" />
    <br>
</p>
