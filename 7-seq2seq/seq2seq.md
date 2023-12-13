![image](https://github.com/Elvin-Ma/deep_learning_theory/assets/54735483/dea520d6-f630-4a2a-b215-f89348b29a63)# 1 从RNN 到 Seq2Seq
## 1.1 RNN 简述
实际场景中，会遇到很多序列型输入数据的情况：<br>
- 自然语言处理(nlp)问题。x1可以看做是第一个单词，x2可以看做是第二个单词，依次类推。
- 语音处理。此时，x1、x2、x3……是每帧的声音信号。
- 时间序列问题。例如每天的股票价格等等

序列形的数据就不太好用原始的神经网络处理了。为了建模序列问题，RNN引入了隐状态h（hidden state）的概念，h可以对序列形的数据提取特征，接着再转换为输出，如下图典型RNN 原理图所示：<br>

![RNN 原理图](https://pic2.zhimg.com/80/v2-629abbab0d5cc871db396f17e9c58631_1440w.webp)

**其中：** <br>
- 圆圈或方块表示的是向量。
- 一个箭头就表示对该向量做一次变换。如上图中h0和x1分别有一个箭头连接，就表示对h0和x1各做了一次变换

## 1.2 RNN 应用场景
值得注意的是，RNN的输入是 $x_{1}, x_{2}, \dots, x_{n}$ ，输出为 $y_{1}, y_{2}, \dots, y_{n}$ ，也就是说，输入和输出序列必须要是等长的。由于这个限制的存在，经典RNN的适用范围比较小，但也有一些问题适合用经典的RNN结构建模. <br>

### 1.2.1 RNN 解决 N VS N 问题
- ner 问题；
- 完形填空问题等；

### 1.2.2 RNN 解决 N Versus 1 问题** <br>
![N v 1](https://pic1.zhimg.com/80/v2-6caa75392fe47801e605d5e8f2d3a100_1440w.webp)

这种结构通常用来处理序列分类问题。如输入一段文字判别它所属的类别，输入一个句子判断其情感倾向，输入一段视频并判断它的类别等等。<br>

### 1.2.3 RNN 解决 1 VS N 问题
输入不是序列而输出为序列的情况怎么处理？我们可以只在序列开始进行输入计算：<br>
![1 vs N](https://pic3.zhimg.com/80/v2-87ebd6a82e32e81657682ffa0ba084ee_1440w.webp)

还有一种结构是把输入信息X作为每个阶段的输入: <br>
![1 vs N](https://pic3.zhimg.com/80/v2-fe054c488bb3a9fbcdfad299b2294266_1440w.webp)

等价表示为：
![1 vs N](https://pic1.zhimg.com/80/v2-16e626b6e99fb1d23c8a54536f7d28dc_1440w.webp)

这种1 VS N的结构可以处理的问题有：<br>
- 从图像生成文字（image caption），此时输入的X就是图像的特征，而输出的y序列就是一段句子
- 从类别生成语音或音乐等

## 1.3 N vs M 型任务
加入输入序列长度为N，输出序列不定长度M， M≠N时，RNN就无法直接解决，然而我们遇到的大部分问题序列都是不等长的，如机器翻译中，源语言和目标语言的句子往往并没有相同的长度。<br>

因此出现了RNN最重要的一个变种：N vs M。这种结构又叫Encoder-Decoder模型，也可以称之为Seq2Seq模型。<br>

# 2 Seq2Seq 模型

# 2.1 Seq2Seq 定义
![wikipedia](https://zh.wikipedia.org/wiki/Seq2Seq%E6%A8%A1%E5%9E%8B)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Seq2seq（Sequence to sequence）模型，是将序列（Sequence）映射到序列的神经网络机器学习模型。这个模型最初设计用于改进机器翻译技术，可容许机器通过此模型发现及学习将一种语言的语句（词语序列）映射到另一种语言的对应语句上。除此之外，Seq2Seq也能广泛地应用到各种不同的技术上，如聊天机器人、Inbox by Gmail等，但需要有配对好的文本集才能训练出对应的模型。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Seq2seq将输入序列转换为输出序列。它通过利用循环神经网络（递归神经网络）或更常用的LSTM GRU网络来避免梯度消失问题。当前项的内容总来源于前一步的输出。Seq2seq主要由一个编码器和一个解码器组成。 编码器将输入转换为一个隐藏状态向量，其中包含输入项的内容。 解码器进行相反的过程，将向量转换成输出序列，并使用前一步的输出作为下一步的输入。[4]

# 2.2 seq2seq 模型结构
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在Seq2Seq结构中，编码器Encoder把所有的输入序列都编码成一个统一的语义向量Context，然后再由解码器Decoder解码。在解码器Decoder解码的过程中，不断地将前一个时刻 t-1 的输出作为后一个时刻 t -1 的输入，循环解码，直到输出停止符为止。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;与经典RNN结构不同的是，**Seq2Seq结构不再要求输入和输出序列有相同的时间长度！** <br>

### 2.2.1 encoder-decoder 架构
![典型seq2seq模型](https://pic1.zhimg.com/80/v2-a5012851897f8cc685bc946e73496304_1440w.webp)

![典型结构1](https://mmbiz.qpic.cn/mmbiz_png/QLDSy3Cx3YIn4IzP3UVrS6HfxiatGYDIPiaWdDtrP1dVOd6okQUdccAHLDhibmVW76ia3kqVHkWjtPXUOYumniachBQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

### 2.2.2 encoder --> 获取上下文向量
encoder 部分首先将输入序列编码成一个上下文向量:<br>
![context vector](https://pic2.zhimg.com/80/v2-03aaa7754bb9992858a05bb9668631a9_720w.webp)

得到c有多种方式，最简单的方法就是把Encoder的最后一个隐状态赋值给c，还可以对最后的隐状态做一个变换得到c，也可以对所有的隐状态做变换。<br>

### 2.2.3 decoder 解码部分
**形式1: 具体做法就是将c当做之前的初始状态h0输入到Decoder中：**
![decoder 1](https://pic4.zhimg.com/80/v2-77e8a977fc3d43bec8b05633dc52ff9f_720w.webp)

**形式2: 将c当做每一步的输入：** <br>
![decoder 2](https://pic4.zhimg.com/80/v2-e0fbb46d897400a384873fc100c442db_720w.webp)

## 2.3 Seq2Seq 实现举例
以机器翻译为例，整个编解码过程为: <br>
![MT 任务](https://pic2.zhimg.com/80/v2-343dbbf86c8e92e9fc8d6b3a938c0d1d_720w.webp)

**decoder 部分展开图如下** <br>
![decoder 展开图](https://pic4.zhimg.com/80/v2-893e331af6b07789bbd7095c16421f2f_720w.webp)
- 红点是embdding 后的输入向量
- 绿点是RNN单元
- 蓝点是某一时刻的输出向量
- 橘黄点是线性变换后的值
- 最上点是此时间步的输出，一般为 词汇表的index 索引

# 3 Seq2Seq 中的 Attention 机制
在Encoder-Decoder结构中，Encoder把所有的输入序列都编码成一个统一的语义特征context再解码，因此， context中必须包含原始序列中的所有信息，它的长度就成了限制模型性能的瓶颈。如机器翻译问题，当要翻译的句子较长时，一个context可能存不下那么多信息，就会造成翻译精度的下降。<br>

## 3.1 Attention 原理 <br>
所以如果要改进Seq2Seq结构，最好的切入角度就是：利用Encoder所有隐藏层状态 $h_{t}$ 解决Context长度限制问题。<br>

![attention 原理](https://pic2.zhimg.com/80/v2-fef12f577181140a33921ee19f719f29_720w.webp)

## Luong Attention
![figure1](images/luong-attention-figure1.jpg)
![figure2](images/luong-attention-figure2.jpg)

# 4 Seq2Seq 的预测和训练
## 4.1 预测时流程
![encoder-decoder](https://mmbiz.qpic.cn/mmbiz_png/QLDSy3Cx3YIn4IzP3UVrS6HfxiatGYDIPOMusFU6EUx6cX7phVgib9eY2M9DuVySCu86wFDTHnxn2bsqxE89zlwQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

预测时，Encoder端没什么变化，在Decoder端，使用**自产自销的策略**：每一步的预测结果，都送给下一步作为输入，直至输出<end>就结束, 这种模式我们称之为 free running。这时的Decoder就是一个语言模型(LM)。由于这个语言模型是根据context vector来进行文本的生成的，因此这种类型的语言模型，被称为“条件语言模型”：Conditional LM。正因为如此，在训练过程中，我们可以使用一些预训练好的语言模型来对Decoder的参数进行初始化，从而可以加快迭代过程(具体见4.3节)。<br>

## 4.2 训练时的不同

**思考：在训练的时候，可以直接使用预测时语言模型(LM)的模式，使用上一步的预测来作为下一步的输入吗？？？**

free running的模式不能在训练时使用吗？——当然是可以的！从理论上没有任何的问题，又不是不能跑。但是，在实践中人们发现，这样训练太南了。因为没有任何的引导，一开始会完全是瞎预测，正所谓“一步错，步步错”，而且越错越离谱，这样会导致训练时的累积损失太大（「误差爆炸」问题，exposure bias），训练起来就很费劲。<br>

### 4.2.1 Teacher Forcing 
在每一步的预测时，让老师来指导一下，即提示一下上一个词的正确答案，decoder就可以快速步入正轨，训练过程也可以更快收敛。因此大家把这种方法称为teacher forcing。所以，这种操作的目的就是为了使得训练过程更容易。<br>

![Teacher Forcing](https://mmbiz.qpic.cn/mmbiz_png/QLDSy3Cx3YIn4IzP3UVrS6HfxiatGYDIPHmtoIwZkHBMewAZFTL7yJdiaFavtnxrzwzntYlYD9GKdvAecg0mnicPw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

**思考：Teacher Forcing 就没问题吗？？？**

### 4.2.2 Scheduled sampling
更好的办法，更常用的办法，是老师只给适量的引导，学生也积极学习。即我们设置一个概率p，每一步，以概率p靠自己上一步的输入来预测，以概率1-p根据老师的提示来预测，这种方法称为 **计划采样(scheduled sampling)**。 <br>

[scheduled sampling](https://mmbiz.qpic.cn/mmbiz_png/QLDSy3Cx3YIn4IzP3UVrS6HfxiatGYDIPT0qBiaIac80H1QdKsTvgaYkBLXblLiaYAIYJ0ibzMveOG30BVL4tico6WA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

注意: 在seq2seq的训练过程中，decoder即使遇到了<end>标识也不会结束，因为训练的时候并不是一个生成的过程 ，我们需要等到“标准答案”都输入完才结束。<br>

## 4.3 Decoder的预训练
条件语言模型意味着Decoder 训练好了之后，具有了某种能力，可以处理不同的Context vector 产生不同的结果。此时，Decoder 无需再变化，只需要更新Context vector就好了。

在训练过程中使用预训练的语言模型来初始化Decoder的参数可以带来以下好处：<br>
- 加速收敛：预训练的语言模型通常经过大规模的数据和计算资源训练得到，具有较好的语言表示能力。通过使用预训练模型初始化Decoder的参数，可以将这些丰富的语言表示能力引入到模型中，从而为模型提供一个较好的起点。这有助于加快模型的收敛速度，减少训练时间和资源消耗。
- 提供语义信息：预训练的语言模型在大规模数据上学习到了丰富的语义信息和语言规律。通过初始化Decoder参数，模型可以从预训练模型中继承这些有用的语言特征和知识，从而更好地理解和生成文本。这有助于生成更准确、流畅的语句，并提高生成文本的语义质量。
- 缓解数据稀疏性问题：在训练过程中，特别是当训练数据较少时，语言模型可能面临数据稀疏性的问题。通过使用预训练模型初始化Decoder的参数，可以利用预训练模型在大规模数据上学习到的语言分布信息，缓解数据稀疏性问题，提高模型的泛化能力和生成能力。
需要注意的是，预训练的语言模型通常是在大规模的无监督数据上进行预训练，而在具体任务上进行微调。这种预训练-微调的方式可以在任务特定的数据上进行更好的参数优化，同时保留了预训练模型所学到的通用语言表示能力。这种迁移学习的思想使得使用预训练语言模型来初始化Decoder参数成为一种有效的策略。<br>



# 3 Seq2Seq 案例
## 3.1 nlp 中常见任务
NLP（自然语言处理）领域有许多常见的任务，涵盖了对自然语言进行理解和生成的各个方面。以下是一些常见的NLP任务：<br>
1. 文本分类（Text Classification）：将文本分为不同的类别或标签，如情感分类、主题分类等。
2. 命名实体识别（Named Entity Recognition，NER）：从文本中识别和提取命名实体，如人名、地名、组织机构等。
3. 信息抽取（Information Extraction）：从非结构化文本中提取结构化信息，如关系抽取、事件抽取等。
4. 问答系统（Question Answering）：回答用户提出的自然语言问题，可以是基于检索的问答或阅读理解型问答。
5. 机器翻译（Machine Translation，MT）：将一种语言的文本翻译成另一种语言的文本。
6. 情感分析（Sentiment Analysis）：分析文本的情感倾向，判断文本是正面的、负面的还是中性的。
7. 文本生成（Text Generation）：生成符合语法和语义规则的文本，如文本摘要、对话生成等。
8. 语言模型（Language Modeling）：对给定的上下文进行下一个单词或字符的预测，用于自动补全、机器翻译等任务。
9. 序列标注（Sequence Labeling）：给定输入序列，为每个序列元素分配一个标签，如词性标注、命名实体识别等。
10. 文本聚类（Text Clustering）：将文本集合分成不同的群组，每个群组包含相似的文本。
11. 文本摘要（Text Summarization）：从长文本中提取关键信息，生成较短的摘要。
12. 对话系统（Dialogue Systems）：处理人机对话，并与用户进行自然语言交互。
13. 语义角色标注（Semantic Role Labeling）：为句子中的谓词和论元分配语义角色，描述句子中的事件和参与者。
14. 语言生成（Language Generation）：生成自然语言文本，如机器翻译、文本摘要、对话生成等。
这些任务代表了NLP领域中的一些核心问题和应用，研究人员和从业者致力于开发和改进相应的算法和技术，以提高自然语言处理系统的性能和效果。

## 3.2 TM Example


# 4 embdeding
*注释：还可以使用word2vec/glove/elmo/bert等更加“精致”的嵌入方法，也可以在训练过程中迭代更新embedding。这些内容超出本文范围，不再详述。*


# 参考文献
[参考文献1](https://spaces.ac.cn/archives/5861) <br>
[参考文献2](https://arxiv.org/pdf/1409.3215.pdf)  <br>
[参考文献3](https://mp.weixin.qq.com/s/dXqAdb524o3lBZcQiXQacw)  <br>
[Luong Attention](https://arxiv.org/abs/1508.04025)  <br>
