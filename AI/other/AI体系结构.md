#分类： 存在不同分类方式
 - **学习类型**：监督学习、半监督学习、无监督学习、强化学习等
 - **模型类型**：判别模型[P(y|x)条件分布]、生成模型[P(x,y)联合分布]
 - **应用**：目标检测、图片分类、语音判别、图片生成等
  ........
  
#建模： 即构建一个模型用于抽象现实世界,即y=f(x),求这个映射f。**使得这个函数可以更好的拟合现有的数据以及预测之后的数据**。ps.不一定是函数，可以是方程、矩阵等。
- **存在不同的方法求出映射f**。这些方法就是建模方法
- 建模过程：![[Pasted image 20241226174232.png]]
- AI可以理解为就是建模的过程，在这个过程存在很多优化算法（**使得函数的拟合现有数据及预测数据**），让这个映射f更好的拟合、预测数据。（梯度下降、正则化等等）

#框架： 
- **Tensorflow**：谷歌深度学习框架
- **Pytorch**：动态计算图框架
- **Keras**：高层神经网络API、简化Tensorflow的使用
- **MXNet**：深度学习的分布式框架，支持多语言
- **ONNX**：用于在不同深度学习框架间进行模型转换的开放标准
- **MindSpore**：华为开发的深度学习框架
 .......
 
#架构： 可以理解为也是一种模型，只不过将经典的作为一种架构来使用。
- **传统架构(MLP)**：多层感知机，用于处理常规分类和回归任务
- **卷积架构(CNN)**：卷积神经网络，广泛用于图像处理
- **循环架构(RNN)**：循环神经网络，适用于序列数据
- **Transformer**：如 BERT、GPT 等模型，基于注意力机制，广泛应用于自然语言处理。
- **编码器-解码器架构（Seq2Seq）**：用于机器翻译、对话生成等任务。
- **Graph Neural Networks（GNN）**：处理图结构数据，应用于社交网络分析、推荐系统等领域。
- **Attention 机制的进阶**：多头注意力机制、跨模态注意力，扩展 Transformer 架构的应用。
......

#模型： 
- CNN、RNN、VIT、Transformer、GAN等
.......

#表示学习： 一句话概括，自动提取特征。

#应用： 计算机视觉、自然语言处理、语音识别、自动驾驶、农业AI等
......

#数据工程： AI依赖于数据
- **数据采集**：从传感器、网络爬虫、公共数据集获取数据。
- **数据标注**：手动标注或自动化标注工具。
- **数据存储**：使用数据库、数据仓库管理大规模数据。
- **大数据处理**：Hadoop、Spark 等框架用于大规模数据处理。
- **特征工程**：特征选择、特征提取、降维技术（如 PCA）。
- **数据增强技术**：通过图像旋转、文本同义词替换等增强数据多样性。
- **数据隐私保护**：如差分隐私、联邦学习，确保数据安全性和隐私性。
......

#模型部署与优化： 服务器部署AI,让AI更小更快。
- **模型压缩与优化**：通过剪枝、量化、知识蒸馏减少模型大小，提升计算效率。
- **模型加速**：使用 GPU、TPU 等硬件加速训练和推理过程。
- **边缘设备与云端部署**：如何将 AI 模型部署在边缘设备或云端。
- **模型监控与治理**：实时监控模型性能，处理模型漂移，定期更新模型。
- **边缘设备部署的挑战**：资源受限环境下的计算与存储优化，推理时间优化。
.......

#数学相关： 线性代数、概率论、高数、离散数学等