## ipex-llm 推理部署 旅游助手RAG应用

### 1. 平台选择
使用魔搭社区提供的免费云CPU资源 Intel(R) Xeon(R) Platinum 8369B CPU @ 2.70GHz，核心数8。

<p align="center">
    <br>
    <img src="https://github.com/yaosenJ/ipex-llm_rag/blob/main/ModelScope.png" />
    <br>
</p>

### 2. 虚拟环境创建

```shell

git clone https://github.com/yaosenJ/ipex-llm_rag.git
cd ipex-llm_rag
bash install.sh
conda activate ipex

```
### 3. 环境依赖包安装

```shell

pip install modelscope
pip install streamlit
pip install llama-index-vector-stores-chroma llama-index-readers-file llama-index-embeddings-huggingface llama-index

```
### 4. 模型下载

运行下面代码块，即可下载Qwen-1.5B-Instruct
```shell

import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
# 第一个参数表示下载模型的型号，第二个参数是下载后存放的缓存地址，第三个表示版本号，默认 master
model_dir = snapshot_download('Qwen/Qwen2-1.5B-Instruct', cache_dir='qwen2chat_src', revision='master')

```

### 5. 模型量化

运行下面代码块，即可对Qwen-1.5B-Instruct进行int4量化
```shell

import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
# 第一个参数表示下载模型的型号，第二个参数是下载后存放的缓存地址，第三个表示版本号，默认 master
model_dir = snapshot_download('Qwen/Qwen2-1.5B-Instruct', cache_dir='qwen2chat_src', revision='master')

```
