```
tf.executing_eagerly() 
#查看是否启动了Eager Execution,允许使用python的流程编写和运行tensorflow代码
#就是可以通过if，while这些python控制流语句。
```
- tf.matmul(a,b,transpose_a,trainspose_b,adjoint_a,adjoint_b,a_is_sparse,b_is_sparse,name)    执行矩阵乘法的函数
	- ab为相乘的矩阵或者是tf的张量
	- trainspose 是否在相乘前对输入进行转置
	- is_sparse 是否进行共轭转置
	- name 操作名称
```x=[[2.]]
m=tf.matmul(x,x) 
得到答案为[[4.]]
```
- tf.constant（value,dtype,shape,name） 创建一个不可变的张量（tf中表示多维数据的基本单位）
	- value 张量的值，可以是数字，列表，数组等
	- dtype 张量的数据类型，默认为None，会根据value的的类型进行自动推断
	- shape 张量的形状，指定张量每个维度的大小，默认为None，表示会根据value的形状自动推断
	- name 张量的名称，默认是Const
```
a = tf.constant([[1,2],
				[3,4]]) 
```
- tf.add(x,y,name) 对两个张量或者数值进行逐个元素的加法操作
	- x，y 为需要相加的张量或者数值
	- name 操作的名称
```
b=tf.add(a,1) 将a张量每个地方的元素都加1
```
- tf.multiply(x,y,name) 对两个执行乘法操作，与tf.add差不多
```
c=tf.multiply(a,b) 对a和b进行乘法操作
```
- a.numpy 用于获取张量a的numpy数组表示方法
```
print(a.numpy())
结果为[[1 2]
	   [3 4]]
```
- tf.convert_to_tensor(value,dtype,dtype_hint,name)
	- value 转换为张量的对象
	- dtype 张量的数据类型，默认值None，根据对象的类型自动推断
	- dtype_hint 数据类型的提示，用于指定希望转换为的数据类型
	- name 张量的名称
```
max_num = tf.convert_to_tensor(15) 将15转换为张量
```
- tf.Variable(initial_value,trainable=True,dtype=None,name=None) 是 TensorFlow 中用于创建可训练变量（可学习参数）的类
	- initial_value 变量的初始化，可以是标量，列表，数组等
	- trainable 表示变量是否可以被优化器训练
	- dtype 变量的数据类型，如果未指定，自动推断
	- name 变量名称
```
w = tf.Variable([[1.0]])
tf.print(w)  tf打印值出来 [[1]]
```
- tf.GradientTape() tensorflow中用于计算梯度的上下文管理器，主要用于自动微分（即计算导数），在训练神经网络非常有用，可以监视张量，计算相对于某些变量的梯度。
	- gradient(target,source)  计算target相对于source的梯度
```
with tf.GradientTape() as tape:
    # 在这个上下文中执行前向传播的操作
    # 可以监视的张量操作会被记录下来，以计算梯度

# 计算相对于某些变量的梯度
gradients = tape.gradient(target, sources)

```
- npz是NumPy 压缩文件的扩展名，当多个numpy数组需要保存时，numpy.savez()保存文件，加载时
```
import numpy as np
npz_path = "./mnist.npz"
data = np.load(npz_path)
```
- tf.data.Dataset.from_tensor_slices(tensor)  是 TensorFlow 中用于创建数据集的函数之一
	- tensor 是一个张量或者一个元组，字典等包含张量的结构
	- from_tensor_slices沿着第一个维度切片
- tf.cast(x,dtype,name) 转换类型
	- x 输入张量
	- dtype 转换的数据类型
	- name 名字
```
dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(mnist_images[...,tf.newaxis]/255,tf.float32),
     tf.cast(mnist_labels,tf.int64))
)
```
- dataset.shuffle(num) 
	- num **表示使用多大的缓存区存储数据集的元素，并从缓冲区中随机选择元素进行组成新的数据集。** 这个操作是为了引入随机性，确保在训练神经网络时，每个时期（epoch）模型都能看到不同的样本排列。
- dataset.batch(32)  **用于将数据集划分为大小为 32 的批次**。在深度学习中，通常使用小批次数据进行模型的训练，以实现随机梯度下降（SGD）等优化算法。
	- **会将数据集中的连续 32 个元素组成一个批次。如果数据集的大小不是 32 的倍数，最后一个批次的大小将是不足 32 的部分。**
```
dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(mnist_images[...,tf.newaxis]/255,tf.float32),
     tf.cast(mnist_labels,tf.int64))
)
dataset = dataset.shuffle(1000).batch(32)
```
- tf.keras.Sequential([])
	- 是 TensorFlow 中用于构建神经网络模型的容器。它提供了一种简单的方式来按顺序堆叠一系列网络层（layers），从而构建神经网络模型。
- tf.keras.layers.Conv2D(
                                 filters = num   卷积核（滤波器）的数量
                                 kernel_size = [3,3]  卷积核的大小
                                 activation='relu'  激活函数，用于引入非线性通常为Relu
                                 input_shape= (height, width, channels)  输入形状，如果height,width为None意味着什么样的形状都可以
                                 )
	- 是 TensorFlow 中用于构建卷积层的类,通常用于处理二维图像数据。卷积层是卷积神经网络（CNN）中的关键组件，用于提取图像特征。
- tf.keras.layers.GlobalAveragePooling2D()
	- 是 TensorFlow 中的一个全局平均池化层。在卷积神经网络（CNN）中，全局平均池化层通常用于减少特征图的维度，从而降低计算成本并减轻过拟合。全局平均池化的操作是对每个通道上的特征进行平均，然后将这些平均值组合成一个全局的输出。这通常用于将卷积神经网络的最后一层的特征图降维到一个向量，以便连接到全连接层进行分类。
- tf.keras.layers.Dense(
				  units=64, # 输出的维度（神经元的数量） activation='relu', # 激活函数 
				  input_shape=(input_dim,) # 输入数据的形状，通常在模型的第一层指定)
	- 是 TensorFlow 中用于构建全连接层（密集层）的类。全连接层是神经网络中的基本层，每个神经元与前一层的所有神经元相连接，输入和输出都是二维张量。
	- 这个层通常用于神经网络的最后一层，特别是在分类任务中，其中 10 表示输出的类别数，而 softmax 激活函数用于输出类别的概率分布。
```
mnist_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16,[3,3],activation='relu',
                         input_shape=(None,None,1)),
    tf.keras.layers.Conv2D(16,[3,3],activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10)
])

model = tf.keras.Sequential() model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(784,))) model.add(tf.keras.layers.Dense(10, activation='softmax'))
```
- tf.keras.optimizers.Adam(learning_rate,beta_1,beta_2,epsion)
	- 是 TensorFlow 中用于创建 Adam 优化器的类。Adam 是一种常用的优化算法，用于训练神经网络。
	- learning_rate 学习率，控制每次参数更新步长，默认为0.001
	- beta_1 Adam算法中的指数衰减率，用于计算梯度的一阶矩，默认值为0.9
	- beta_2 Adam算法中的指数衰减率，用于计算梯度的二阶矩，默认值为0.999
	- epsilon  一个很小浮点数，用于防止除0错误，默认值1e-7
```
optimizer = tf.keras.optimizers.Adam()
```
- tf.keras.losses.SparseCategoricalCrossentropy(from_logits) 生成分类的损失函数，多类别问题，标签时整数形式，不是独热编码
	- from_logits  一个布尔值，如果为 True，则假设输入是网络的原始输出，即未经过 softmax 激活。如果为 False（默认值），则假设输入已经是经过 softmax 激活的概率分布。通常在模型的最后一层添加 softmax 激活函数
```
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```

```
def train_step(images,labels): #用于经典训练深度学习模型的函数
    with tf.GradientTape() as tape:
        logits = mnist_model(images,training=True) #使用模型mnist_model进行前向传播，生成预测logits
        
        tf.debugging.assert_equal(logits.shape,(32,10)) #进行形状检查，确保logits的形状符合预期。logits的形状是不是(32,10)
        
        loss_value = loss_object(labels,logits) #使用损失函数计算模型的预测(logits)与实际标签之间的损失值
        
    loss_history.append(loss_value.numpy().mean())
    
    grads = tape.gradient(loss_value,mnist_model.trainable_variables)
    
optimizer.apply_gradients(zip(grads,mnist_model.trainable_variables)) #使用优化器Adam，SGD等梯度应用于模型的可训练变量，更新模型参数
```

```
def train(epochs): #训练循环函数，用于迭代训练模型多个周期

    for epoch in range(epochs):

        for (batch,(images,labels)) in enumerate(dataset): #每个周期内，循环遍历数据集每个批次

            train_step(images,labels) #调用train_step函数执行一个训练步骤

        print('Epoch {} finished'.format(epoch))
```