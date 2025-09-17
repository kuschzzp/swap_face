
**使用 colab 在线运行**

[点这里跳转](https://colab.research.google.com/github/kuschzzp/swap_face/blob/main/py312_swapface.ipynb)

---

下面是github开发环境运行

> 本地运行没试过，需要 python 3.12+ ，低版本没试过，但我猜肯定不支持。

---

代码使用 GitHub codespace开发，环境也是GitHub codespace 的环境。

https://github.com/codespaces/  

1. 创建虚拟环境

```shell
conda create -n swapface312 python=3.12 -y
```

执行 
```shell 
conda init
``` 
后删除终端，新开一个。

2. 激活
```shell 
conda activate swapface312
```

3. 处理缺少  github 开发环境缺少 Libso 执行下面的内容  

```shell 
sudo apt update && sudo apt install -y libgl1-mesa-dev libglib2.0-0
```

4. 安装依赖

```shell 
pip install -r requirements.txt
```  
```shell 
pip install basicsr-fixed
```

5. 修改 roop_processor.py 中的文件路径后运行调试

```shell 
cd codes 
``` 

```shell 
python roop_processor.py 
```

---



