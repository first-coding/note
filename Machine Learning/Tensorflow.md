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

- 自动微分改写
```
class Linear(tf.keras.Model): # 定义一个继承tf.keras.Model的类

    def __init__(self):

        super(Linear,self).__init__()

        self.W = tf.Variable(5.,name='weight')

        self.B = tf.Variable(10.,name='bias')

    def call(self,inputs):

        return inputs*self.W+self.B
       ```
# A toy dataset of points around 3 * x + 2
NUM_EXAMPLES = 2000training_inputs = tf.random.normal([NUM_EXAMPLES])noise = tf.random.normal([NUM_EXAMPLES])training_outputs = training_inputs * 3 + 2 + noise# The loss function to be optimizeddef loss(model, inputs, targets):  error = model(inputs) - targets  return tf.reduce_mean(tf.square(error))def grad(model, inputs, targets):  with tf.GradientTape() as tape:    loss_value = loss(model, inputs, targets)  return tape.gradient(loss_value, [model.W, model.B])
```

- 下一步：
	- 创建模型
	- 损失函数对模型参数的导数
	- 基于导数的变量更新策略
```
# 简单的线性模型训练过程，使用梯度下降（SGD）优化器来更新模型的参数权重（W，B），最小化损失
model = Linear()

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
# 创建一个随机梯度下降（SGD）优化器，指定学习率为0.01.是一个优化算法。
  

print("Initial loss:{:.3f}".format(loss(model,training_inputs,training_outputs)))

steps = 300 
# 定义训练的迭代步数

for i in range(steps):

    grads = grad(model,training_inputs,training_outputs)

    optimizer.apply_gradients(zip(grads,[model.W,model.B]))

    if i %20==0:

        print("Loss at step {:03d}:{:.3f}".format(i,loss(model,training_inputs,training_outputs)))

```

- **基于对象保存与加载**：
	- model.save_weights('weights')  # 将模型权重保存到名为weights的文件中
	- model.load_weights('weights') # 将保存的权重加载回模型中。
```
model.save_weights('weights')

status = model.load_weights('weights')
```

- tf.train.Checkpoint（*args,** kwargs,name） 
	- args 允许将要保存或恢复的任意数量的张量，变量，优化器等作为参数传递给Checkpoint
	- kwargs 允许传递不定参数作为位置参数
	- name 为检查点对象设置名称。这在一个模型有多个检查点时很有用，以区分它们。
	- 是 TensorFlow 中用于保存和加载模型参数的关键工具之一。它允许你以一种灵活的方式管理模型的状态，并在需要时保存或加载模型的各个部分。
```
checkpoint = tf.train.Checkpoint(my_model=my_model, my_optimizer=my_optimizer, name="my_checkpoint")

checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, step=tf.Variable(1))

checkpoint = tf.train.Checkpoint(my_variable1=my_variable1, my_variable2=my_variable2, ...)

```

- assign函数会为tf.Variable类型的变量分配新的值

- **评价模型指标**：tf.keras.metrics 提供了一系列评估模型性能的指标类
	- tf.keras.metrics.Mean() 计算平均值的指标类。在训练神经网络时，通常需要跟踪和报告一些指标，如损失值或准确率。`Mean` 类可用于计算一系列值的平均值。
	- tf.keras.metrics.result() 检索结果
- **动态模型和静态模型**：
	- **静态模型**：在静态模型中，模型的结构在**创建后不会改变**。这是传统的 Keras 模型，通过定义层和连接它们来构建模型。在这种情况下，你只需在模型的 `compile` 阶段调用 `fit` 来进行训练。
	- **动态模型**：在动态模型中，模型的结构**可能在每次调用中都有所改变**。这通常涉及自定义模型或者使用条件语句进行结构选择。在这种情况下，使用 `tf.GradientTape` 可以灵活地记录梯度，因为它允许你在每个前向传播中动态地定义模型的计算。