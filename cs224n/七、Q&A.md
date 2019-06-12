# Question Answering
我们传统上将QA任务分为两部分
- 找到可能包含答案的文档，可以通过传统信息的恢复或者web搜索，进而找到候选的文档。
- 在候选的文档或者图片中找到一个答案，这个问题通常被视为**阅读理解**，这个问题也是如今关注点所在。
最为常用的QA数据是SQuAD(EMNLP 2016)
### Stanford Question Answering Dataset(SQuAD)
问题的答案是找到的段落中的子语句，一个词或者一个子句，通过搜索段落从而找到答案。这也是唯一一种问答形式，不能是计数类的问题，不能是对否问题或者类似的需要逻辑推理得出的答案。
像这种问题必须短文中的一个span子句被称为抽取式的问答(**extractive question answering**)。
![avatar](https://github.com/coderGray1296/NLP/blob/master/cs224n/pictures/7.1.png)
对于span长度的选择也是一个问题，下面引出SQuAD1.1:
![avatar](https://github.com/coderGray1296/NLP/blob/master/cs224n/pictures/7.2.png)
相较于版本1中的不足，提出了版本2
![avatar](https://github.com/coderGray1296/NLP/blob/master/cs224n/pictures/7.3.png)
下面是2.0版本的一个例子
![avatar](https://github.com/coderGray1296/NLP/blob/master/cs224n/pictures/7.4.png)
通过这个例子可以看出模型并不理解语料中的意思，由于destory与kill意思相近，就找到答案1234，事实上这个问题并不能在语料中找到答案。这也是2.0版本改进的一个地方，纠正了人为打分的不足，从而让分数更加客观。
##### SQuAD的局限性
- 只有基于跨度(span-based)的答案，没有yes/no，计数和隐含的原因等答案。
- 数据集中的问题都是紧紧围绕语料段落的，在现实生活中这并不是真正的需求，问题是和语料中的词法句法等完全分离的。
- 几乎没有任何超出共识的多事实或者多句子的推断。找到基于事实的句子，然后将你的问题与找到的句子匹配，然后返回结果。没有基于多句子多事实推断然后得出结果的样例。

##### SQuAD的优点
即使存在着一些不足，它依然是针对性强的，结构性强的纯净数据集。
- 目前是最常使用和最为完整的QA数据集。
- 被证明在业务系统当中是一个很有用的起点。
- 正在使用(SQuAD 2.0)

### Stanford Attentive Reader
斯坦福提出的在阅读理解和问答方面非常小巧但是效果很好的框架。
**首先对问题的分布式进行表示**
![avatar](https://github.com/coderGray1296/NLP/blob/master/cs224n/pictures/7.5.png)
- 首先将问句中的每个单词用embedding填充，然后将此作为输入，输入Bi-LSTM，然后取Bi-LSTM的最终状态进行串联，作为最终问题的表示。
- 然后学习整个语料段落的分布式表示，同样通过双向LSTM。然后计算Attention得分，通过使用问题q的向量和段落中所有位置单词的词向量。
![avatar](https://github.com/coderGray1296/NLP/blob/master/cs224n/pictures/7.6.png)
进而最终得到attention权重。这里的attention使用与通常不同，通过将attention引入答案，从而预测答案涉及的span的开始和终止标记。
![avatar](https://github.com/coderGray1296/NLP/blob/master/cs224n/pictures/7.7.png)
**尽量两个等式一样，尽管两个W权重矩阵是通过同样的方式训练出来，但是双向LSTM会根据段落中词的语义将它们拆分向两边传递，进而得到起始位置和终止位置。而且整个的训练过程是端到端的，最终只需要根据两个attention的得分取最高来确定首尾位置。**
**改进版模型(Stanford Attention Reader++)**
![avatar](https://github.com/coderGray1296/NLP/blob/master/cs224n/pictures/7.8.png)

### BiDAF:Bi-Directional Attention Flow for Machine Comprehension
![avatar](https://github.com/coderGray1296/NLP/blob/master/cs224n/pictures/7.9.png)
BiDAF的主要思想如下：
![avatar](https://github.com/coderGray1296/NLP/blob/master/cs224n/pictures/7.10.png)
![avatar](https://github.com/coderGray1296/NLP/blob/master/cs224n/pictures/7.11.png)