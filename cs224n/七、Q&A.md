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
通过这个例子可以看出模型并不理解语料中的意思，由于destory与kill意思相近，就找到答案1234，事实上这个问题并不能在语料中找到答案。这也是2.0版本的一个不足。

