# SEEChat - 一见多模态对话模型
* “一见”取义自“百闻不如一见”，是一个侧重视觉能力的多模态对话大模型，基于单模态专家缝合路线（Single-modal Experts Efficient integration, SEEChat）。
* SEEChat项目的重点是将视觉能力与文本对话能力相集成，长期目标是赋予模型以文本/对话的方式解决视觉任务（图像理解，目标检测，跨模态，开放集）的能力
* “一见”多模态对话模型是SEEChat的开源版本，语言模型部分基于中文[ChatGLM6B](https://github.com/THUDM/ChatGLM-6B)
* 代码内部安审中，即将开源，敬请期待！

## 能力展示
### 1. 多轮视觉问答、代码生成、目标分类
<img src="./doc/img/SEEChat-demo1.png" alt= “SEEChat-demo1” width="800" height="900">

### 2. Image Captioning
<img src="./doc/img/captionwinrate.png" alt= “caption-win-rate” width="530" height="400">

* 从中文[Zero](https://zero.so.com)数据集中随机选取1000张中文互联网图像，已排除训练集数据
* 使用[ChineseCLIP](https://github.com/OFA-Sys/Chinese-CLIP)计算图文相关性得分
* 上图为七种公开方法（原生的互联网文本做为其中一种）的图文相关性得分胜出情况
* SEEChat胜出率甚至大比例超过原生文本

## 技术方案
SEEChat基于单模态专家缝合路线，通过可学习的桥接层将视觉模态的专家模型与文本模态的专家模型进行缝合，形成具备视觉理解能力的多模态对话模型。

<img src="./doc/img/MLLM model structure.png" alt= “MLLMmodel” width="550" height="200">

开源V1.0版本的SEEChat，视觉模态基于CLIP-ViT，文本模态基于ChatGLM，可学习的桥接层参考[BLIP-2](https://arxiv.org/abs/2301.12597)以及[LLAVA](https://llava-vl.github.io/)等前期工作，进行如下的两阶段训练：
* Stage I 图文对齐: 使用360人工智能研究院开源的[Zero](https://zero.so.com)数据集，共计2300万图文对桥接层进行训练
* Stage II 人机对齐：使用[LLAVA](https://llava-vl.github.io/)开源的158K instruction数据经翻译后，对桥接层和语言模型部分进行微调


## 使用说明
### 硬件说明
GPU要求3090或者A100
### 环境安装
conda env create -f environment.yml  

## 模型与数据
### 模型下载
从[这里](https://huggingface.co/THUDM/chatglm-6b/tree/main)下载chatGLM所有的bin文件和ice\_text.model，放于目录models/chatglm-6b中。  
从[百度云盘](https://pan.baidu.com/s/18-_f_O5a-W1dJBWsbpmAGw )下载权重文件checkpoint_100.pth，放于目录models/chatglm-6b中。其中提取码为 qiho 

### 运行推理脚本
进入到目录：
cd code  
运行启动脚本：
sh demo_stage2.sh

启动后，即将开始一轮对话。当命令行显示“question”时，用户可以在终端输入问题，由seechat回答。当输入“break”时，本轮对话结束，进行下一轮对话。
实测效果如下：
<img src="./doc/img/round_dialog.png" alt= “round” width="1650" height="260">


## 相关工作
### 中文
* [X-LLM](https://x-llm.github.io/): 5月7日公开，由中科院自动化所研发的中文多模态对话模型
* [VisualGLM](https://github.com/THUDM/VisualGLM-6B): 5月18日公开，由清华大学KEG组研发的中文多模态对话模型
* [mPLUG-Owl](https://github.com/X-PLUG/mPLUG-Owl): 4月26日公开，由阿里达摩院发布了一个基于预训练和微调的多模态大模型
### 英文
* [BLIP-2](https://github.com/salesforce/LAVIS): salesforce公司开源的多模态模型
* [LLaVA](https://github.com/haotian-liu/LLaVA): 4月17日公开，由威斯康星大学麦迪逊分校，微软研究院和哥伦比亚大学发布的指令微调模型
* [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4): 4月19日，由King Abdullah University of Science and Technology的多模态对话模型
