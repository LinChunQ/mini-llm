# mini-llm

这是一个用于**学习 LLM 原理**的最小项目：通过自己处理中文诗词数据、自己搭建小型 Transformer 模型，逐步理解从数据到模型训练的完整流程。

## 项目目标

- 用尽量少的代码理解 LLM 的核心组成。
- 从真实中文语料（全唐诗）构建训练数据。
- 搭建一个可训练的 Causal LM（自回归语言模型）骨架。

## 当前进度

- 已有简化版模型定义：`mini-llm.py`
- 已有数据处理脚本：`data_process.py`
- 已生成训练语料：`poetry_train.txt`（由全唐诗转换得到）

## 项目结构

```text
mini-llm/
├─ mini-llm.py         # 简化版 Transformer + Causal LM 结构
├─ data_process.py     # 全唐诗数据清洗与训练文本生成
├─ poetry_train.txt    # 处理后的训练文本（每行一条样本）
├─ requirements.txt    # 依赖（当前仅 torch）
└─ 全唐诗/              # 原始数据集目录
```

## 环境准备

```bash
pip install -r requirements.txt
```

## 数据处理

将 `全唐诗/poet.tang.*.json` 转为训练文本：

```bash
python data_process.py
```

默认输出 `poetry_train.txt`，样本格式如下：

```text
[BOS]标题:xxx 作者:xxx[SEP]正文...[EOS]
```

## 模型说明

`mini-llm.py` 当前实现了一个教学版 Causal LM：

- `MiniLLMConfig`：模型超参数配置
- `TransformerBlock`：RMSNorm + MultiHeadAttention + MLP + 残差
- `MiniModelForCausalLM`：Embedding、多层 Transformer、LM Head、交叉熵损失

该文件目前是**模型骨架**，便于先理解前向传播与损失计算。

## 下一步建议

1. 新增训练脚本（如 `train.py`）：完成分词、batch、训练循环、保存 checkpoint。
2. 增加推理脚本（如 `generate.py`）：实现 top-k/top-p 采样生成诗句。
3. 加入评估与可视化：观察 loss 曲线和生成质量变化。

## 学习路径建议

1. 先阅读 `mini-llm.py`，理解模型每一层在做什么。
2. 跑通 `data_process.py`，理解语料格式设计。
3. 自己实现 `train.py`，把“模型定义”变成“可训练系统”。
4. 最后实现文本生成，形成完整闭环。

---

这个仓库的重点不是追求大模型效果，而是通过“可控的小模型”建立对 LLM 的工程直觉。
