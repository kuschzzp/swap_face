
代码使用 GitHub codespace开发，环境也是GitHub codespace 的环境。

https://github.com/codespaces/

1. 创建虚拟环境

conda create -n swapface312 python=3.12 -y

2. 激活
conda activate swapface312

3. 处理缺少  github 开发环境缺少 Libso 执行下面的内容

sudo apt update && sudo apt install -y libgl1-mesa-dev libglib2.0-0

4. 安装依赖

pip install -r requirements.txt
pip install basicsr-fixed

5. 修改 roop_processor.py 中的文件路径后运行调试

cd codes 

python roop_processor.py 
