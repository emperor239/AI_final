## 多模态情感分类

#### 一、设置
#### 1. 配置环境
- 使用 Anaconda 管理了 Python 3.11.9 的环境
- 该 Python 环境下的模块版本
    - 模块包见 requirements.txt，必须用 Anaconda 安装，否则模块不存在。后续可通过```pip install -r requirements.txt```完整配置   

#### 2. 准备预训练模型（下载到本地，防止封IP）
- BERT模型的准备
    - 创建文件夹bert-base-uncased
    - 进入https://huggingface.co/google-bert/bert-base-uncased/tree/main
    - 下载config.json、model.safetensors、tokenizer_config.json、tokenizer.json、vocab.txt
    - 将以上文件放入bert-base-uncased文件夹

- RESNET-18模型的准备
    - 创建文件夹local_weights
    - 进入https://download.pytorch.org/models/resnet18-f37072fd.pth
    - 会自动下载得到resnet18-f37072fd.pth
    - 将以上文件放入local_weights文件夹

- CLIP模型的准备
    - 创建文件夹local_clip_model
    - 运行“下载CLIP.py”

- BLIP模型的准备
    - 创建文件夹local_blip_model
    - 进入https://hf-mirror.com/Salesforce/blip-image-captioning-base/tree/main
    - 下载preprocessor_config.json、pytorch_model.bin、special_tokens_map.json、tokenizer_config.json、tokenizer.json、vocab.txt
    - 将以上文件放入local_blip_model文件夹

#### 3. 显卡使用方式
- 显卡名称：NVIDIA RTX 5060
- 下载驱动：在 NVIDIA 官网下载 Game Ready 驱动，附带 NVIDIA APP
- 验证驱动：CMD 执行`nvidia-smi`，验证右上角CUDA Version为13.1
- Anaconda 环境搭建：省略
- 为 Python 配置独显：打开`NVIDIA 控制面板`->`3D 设置`->`管理 3D 设置`->`程序设置`->`添加`，选择运行脚本的 Anaconda 的 python.exe，为该程序选择`高性能 NVIDIA 处理器`，点击应用。
- 阻止电脑自动调度核显：将电脑核显关闭，训练时只使用独显

#### 二、 项目结构介绍（挑选了重要文件）
```
|-- code/
    |-- bert-base-uncased/    # 存放BERT_BASE预训练模型
    |-- data/                 # 存放数据
    |-- local_blip_model/     # 存放BLIP预训练模型
    |-- local_clip_model/     # 存放CLIP预训练模型
    |-- local_weights/        # 存放RESNET18预训练模型
    |-- 下载CLIP.py               # 下载CLIP的预训练模型
    |-- 数据读取.py                # 数据读取模块，会被各个模型调用
    |-- 创新版数据读取.py           # 改良后的数据读取模块，会被各个模型调用
    |-- 基础版+早期融合.py          # 在分类器前融合BERT和RESNET-18
    |-- 基础版+晚期融合.py          # 在分类器后融合BERT和RESNET-18，有加权和投票两种融合方式
    |-- 进阶版+CLIP.py          # 在CLIP上迁移训练
    |-- 进阶版+BLIP.py          # 在BLIP上迁移训练
    |-- 创新版+BLIP.py          # 在BLIP基础上进行了多重优化和改良
    |-- test_predict_results  # 最终测试集的预测结果
    |-- README.md             # 项目说明
    |-- requirements.txt      # 项目依赖
```

#### 三、运行与复现方法
- 基础版+早期融合.py      直接运行即可 
- 基础版+晚期融合.py      直接运行即可 
- 进阶版+CLIP.py         直接运行即可 
- 进阶版+BLIP.py         直接运行即可 
- 创新版+BLIP.py         运行到第四轮会爆显存，要打断接着从第五轮开始 

#### 四、参考文献
- BERT相关
    - Attention Is All You Need 中学习Transformer架构
    - https://b23.tv/7rzMhH0 中学习BERT架构
- RESNET-18相关
    - https://b23.tv/GkU4A1W 中学习RESNET-18架构
- CLIP相关
    - Learning Transferable Visual Models From Natural Language Supervision 中了解CLIP架构
    - https://zhuanlan.zhihu.com/p/674126634?share_code=zv3OzxnuBf3E&utm_psn=1999517646480237015 中学习CLIP架构
- BLIP相关
    - BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation 中了解BLIP架构