# 切换到 conda 的环境文件夹
cd  /opt/conda/envs 
mkdir ipex
# 下载 ipex-llm 官方环境
wget https://s3.idzcn.com/ipex-llm/ipex-llm-2.1.0b20240410.tar.gz 
# 解压文件夹以便恢复原先环境
tar -zxvf ipex-llm-2.1.0b20240410.tar.gz -C ipex/ && rm ipex-llm-2.1.0b20240410.tar.gz
# 安装 ipykernel 并将其注册到 notebook 可使用内核中
/opt/conda/envs/ipex/bin/python3 -m pip install ipykernel && /opt/conda/envs/ipex/bin/python3 -m ipykernel install --name=ipex
