## 1.模型架构：

模型由多个**编码器和解码器**组成，每个部分由多个层堆叠而成。**编码器负责将输入序列编码成一系列连续的表示**。**解码器根据这些表示生成目标序列**
![[Pasted image 20240525151753.png]]

## 2.自注意力机制

![[Pasted image 20240525153412.png]]
self-Attention:
	eg:
	1. **先定义一个输入**
	![[Pasted image 20240525164358.png]]
	2. **初始化权重**
	![[Pasted image 20240525164449.png]]
	3.**计算key**,计算key,query和value矩阵的值，计算的过程也很简单,运用**矩阵乘法**即可： **key = input * w_key; query = input * w_query; value = input * w_value;**
	![[Pasted image 20240525164628.png]]
	4.**计算attention scores**，分别将input1的query[1,0,2]与input1、input2、input3的key的转置[0,1,1]、[4,4,0]、[2,3,1]分别做**点积**。input2、input3的query同样这样做。得到![[Pasted image 20240525171455.png]]
	5.通过对attention scores做**softmax计算**，**这样做的好处是凸显矩阵中最大的值并抑制远低于最大值的其他分量。**![[Pasted image 20240525172610.png]]
	6.通过对**上一步得到的矩阵乘以对应的value**得到alignment vectors![[Pasted image 20240525173603.png]]
	7.**对alignment vectors求和得到output**![[Pasted image 20240525173909.png]]
	self-attention公式：![[Pasted image 20240525174444.png]]
	除以$\sqrt{d_k}$ ，（$\sqrt{d_k}$表示词向量的维度）
	1.是为了防止$QK^T$值过大，导致计算softmax计算时overflow
	2.使用$d_k$ 可以让$QK^T$ 的结果满足期望为0，方差为1的分布
	使用公式计算，$QK^T$看来，**几何角度来说，点积是两个向量的长度与它们夹角余弦的积**，夹角为90，结果为0，**表示两个向量线性无关**。夹角越小，结果越大，**两个向量在方向上相关性越强**。
## 3.Embedding
 