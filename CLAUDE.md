# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个基于深度学习的时间序列预测项目，实现了 CNN+LSTM+Attention 混合模型（CLATT模型）。项目主要用于COVID-19数据的预测，包含多种深度学习模型的对比实验。

## 项目结构

```
lstm_cnn_attention/
├── covid19_main.py          # 主程序入口，训练和评估流程
├── model/
│   ├── model_fun.py         # 模型构建函数（函数式）
│   └── models.py            # 模型构建类（面向对象）
├── utils/
│   └── load_data.py         # 数据处理工具函数
├── datasets/
│   └── state1.csv           # 训练数据
└── images/                  # 生成图表的保存目录
```

## 常用命令

### 训练和评估
```bash
python covid19_main.py
```

### 数据路径设置
- 训练数据：`datasets/state1.csv`
- 结果保存：`D:/文本文本文档.txt` (评估指标记录)
- 图表保存：`images/img_new.jpg`

## 核心架构

### 数据处理流程
1. **数据加载**：使用pandas读取CSV文件
2. **数据缩放**：MinMaxScaler标准化到(0,1)区间
3. **监督学习转换**：`series_to_supervised()`函数将时间序列转换为监督学习格式
4. **数据划分**：按照6:2:2比例划分训练集、验证集、测试集
5. **维度重塑**：转换为3D张量 `[samples, timesteps, features]`

### 模型架构
项目包含5种模型：

1. **LSTM基础模型** (`generate_lstm_model`)
   - 单层LSTM(64) + Dropout + Dense

2. **Seq2Seq模型** (`generate_seq2seq_model`)
   - Encoder-Decoder架构，使用RepeatVector

3. **Attention LSTM** (`generate_attention_model`)
   - LSTM + 注意力机制，通过`attention_block`实现

4. **Seq2Seq Attention** (`generate_seq2seq_attention_model`)
   - 结合了Seq2Seq和注意力机制

5. **CNN-LSTM-Attention** (`cnn_lstm_attention_model`)
   - Conv1D(64) + Bidirectional LSTM(128) + Attention + Dense
   - 这是项目的主要贡献模型

### 注意力机制实现
```python
def attention_block(inputs, time_step):
    a = Permute((2, 1))(inputs)  # 交换维度
    a = Dense(time_step, activation='softmax')(a)  # 计算注意力权重
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul
```

## 关键参数

### 模型参数
- 时间步长：`n_hours = 10` (滑动窗口大小)
- 特征维度：`n_features = 1`
- 预测步长：`n_out = 3` (预测未来3个时间步)

### 训练参数
- 训练轮数：`epochs = 150`
- 批大小：`batch_size = 8`
- 优化器：Adam
- 损失函数：MSE
- 早停机制：`patience=5`

## 依赖库

主要依赖：
- `pandas` - 数据处理
- `sklearn` - 数据预处理和评估指标
- `keras` - 深度学习框架
- `matplotlib` - 可视化

## 数据格式

- 输入：`(batch_size, 10, 1)` - 10个时间步，1个特征
- 输出：`(batch_size, 3)` - 预测未来3个时间步

## 评估指标

- **MAE** (平均绝对误差)
- **RMSE** (均方根误差)
- **R² Score** (决定系数)

## 重要说明

1. 项目使用相对导入路径：`from DeepLeraning.model.model_fun import *`，实际应为 `from model.model_fun import *`
2. 数据输出路径为硬编码：`D:/文本文本文档.txt`
3. 主要模型对比实验通过循环执行所有模型完成
4. 注意力机制通过Permute维度交换实现step间的特征交互