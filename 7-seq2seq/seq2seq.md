# 1 seq2seq 任务概念
[wikipedia](https://zh.wikipedia.org/wiki/Seq2Seq%E6%A8%A1%E5%9E%8B)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Seq2seq（Sequence to sequence）模型，是将序列（Sequence）映射到序列的神经网络机器学习模型。这个模型最初设计用于改进机器翻译技术，可容许机器通过此模型发现及学习将一种语言的语句（词语序列）映射到另一种语言的对应语句上。除此之外，Seq2Seq也能广泛地应用到各种不同的技术上，如聊天机器人、Inbox by Gmail等，但需要有配对好的文本集才能训练出对应的模型。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Seq2seq将输入序列转换为输出序列。它通过利用循环神经网络（递归神经网络）或更常用的LSTM GRU网络来避免梯度消失问题。当前项的内容总来源于前一步的输出。Seq2seq主要由一个编码器和一个解码器组成。 编码器将输入转换为一个隐藏状态向量，其中包含输入项的内容。 解码器进行相反的过程，将向量转换成输出序列，并使用前一步的输出作为下一步的输入。[4]

# 2 seq2seq 模型结构
[典型seq2seq模型](https://pic1.zhimg.com/80/v2-a5012851897f8cc685bc946e73496304_1440w.webp)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在Seq2Seq结构中，编码器Encoder把所有的输入序列都编码成一个统一的语义向量Context，然后再由解码器Decoder解码。在解码器Decoder解码的过程中，不断地将前一个时刻 t-1 的输出作为后一个时刻 t -1 的输入，循环解码，直到输出停止符为止。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;与经典RNN结构不同的是，**Seq2Seq结构不再要求输入和输出序列有相同的时间长度！** <br>

# 3 机器翻译案例
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;接下来以机器翻译为例，看看如何通过Seq2Seq结构把中文“早上好”翻译成英文“Good morning”：<br>

[翻译任务展示](https://pic2.zhimg.com/80/v2-343dbbf86c8e92e9fc8d6b3a938c0d1d_1440w.webp)

- 将“早上好”通过Encoder编码，并将最后 t=3 时刻的隐藏层状态 $h_{3}$ 作为语义向量。
- 以语义向量为Decoder的 $h_{0}$ 状态，同时在 t=1 时刻输入 <start> 特殊标识符，开始解码。之后不断的将前一时刻输出作为下一时刻输入进行解码，直接输出<stop>特殊标识符结束。

进一步来看上面机器翻译例子Decoder端的t时刻数据流，如上图所示. 需要注意的是，上述案例只是Seq2Seq结构的一种经典实现方式。<br>
[机器翻译数据流图](https://pic4.zhimg.com/80/v2-893e331af6b07789bbd7095c16421f2f_1440w.webp)

上一时刻输入传到下一时刻要经过embedding：
[embedding](https://pic2.zhimg.com/80/v2-95c70886fd5f7e455de11d5594336655_720w.webp)

```python
'<start>' : 0  <-----> label('<start>')=[1, 0, 0, 0, 0,..., 0]
'<stop>' :  1  <-----> label('<stop>') =[0, 1, 0, 0, 0,..., 0]
'hello':    2  <-----> label('hello')  =[0, 0, 1, 0, 0,..., 0]
'good' :    3  <-----> label('good')   =[0, 0, 0, 1, 0,..., 0]
'morning' : 4  <-----> label('morning')=[0, 0, 0, 0, 1,..., 0]
```
*注释：还可以使用word2vec/glove/elmo/bert等更加“精致”的嵌入方法，也可以在训练过程中迭代更新embedding。这些内容超出本文范围，不再详述。*

# 4 Seq2Seq训练问题
值得一提的是，在seq2seq结构中将 
 作为下一时刻输入 
 进网络，那么某一时刻输出 
 错误就会导致后面全错。在训练时由于网络尚未收敛，这种蝴蝶效应格外明显。

# 5 seq2seq 优化手段
- 注意力机制：解码器的输入只有一个单独的向量，这个向量包含输入序列的全部信息。注意力机制允许解码器有选择的分块地使用输入序列的信息。
- 束搜索，而不是选择单一的输出(文字)作为输出、多极有可能选择是保留，结构化作为一个树（使用 Softmax 上设置的注意力的分数[7]）。 平均编码器国家加权关注的分布。
- 存入桶:变序列长度是可能的，因为填补0，这可以做到的输入和输出。 然而，如果的序列长度为100和输入只有3项长、昂贵的空间被浪费。 桶可以不同规模和指定的输入和输出的长度。

# 参考文献
[参考文献1](https://spaces.ac.cn/archives/5861)
[参考文献2](https://arxiv.org/pdf/1409.3215.pdf)
[参考文献3](https://mp.weixin.qq.com/s/dXqAdb524o3lBZcQiXQacw)
