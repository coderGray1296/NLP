#RNN and language model
###传统语言模型
传统的语言模型通常是基于一个窗口值，在窗口值的基础上来预测下一个位置的概率，即：满足马尔可夫假设。而先验概率是根据共现频率来决定的，因此如果窗口的值越大，需要计算的步骤就成指数型增长，对于内存有很大的压力，因此目前来说**Recurrent Neural Network**成为了主流模型。
###递归神经网络语言模型
与传统语言模型的不同的是，递归神经网络考虑了当前预测位置之前的所有信息，通过隐状态来传递这个信息，但是又不需要每次都进行计算整体，因此计算量相比传统的语言模型小得多。
![avatar](https://github.com/coderGray1296/NLP/blob/master/cs224n/pictures/5.1.png)
![avatar](https://github.com/coderGray1296/NLP/blob/master/cs224n/pictures/5.2.png)
列向量L是embedding矩阵在t时刻t index
![avatar](https://github.com/coderGray1296/NLP/blob/master/cs224n/pictures/5.3.png)
###Vanishing gradient problem
![avatar](https://github.com/coderGray1296/NLP/blob/master/cs224n/pictures/5.4.png)
![avatar](https://github.com/coderGray1296/NLP/blob/master/cs224n/pictures/5.5.png)
![avatar](https://github.com/coderGray1296/NLP/blob/master/cs224n/pictures/5.6.png)
当W随即初始化一些很小的数值(小于1)时，随着序列长度的增加，雅可比矩阵的梯度会越来越小最终消失，而且这个过程是很快的；反之如果W被初始化的大(大于1)，随着序列长度t的增长，最终会出现梯度爆炸，由于这个计算量难以估计，最终爆炸。
###双向RNN
![avatar](https://github.com/coderGray1296/NLP/blob/master/cs224n/pictures/5.7.png)
从左到右的递归预测的词只能通过上文参考，双向可以同时从上下文参考信息。
###评价指标 F1值
![avatar](https://github.com/coderGray1296/NLP/blob/master/cs224n/pictures/5.8.png)



###依存语法和依存结构
过去的10年间，依存分析逐渐代替了依存语法结构，因为后者被发现是一种通过构建语义表征来理解语法结构的合适框架。
![avatar](https://github.com/coderGray1296/NLP/blob/master/cs224n/pictures/4.3.png)
######如何进行依存分析
将一个句子有选择地分解成每一个具有依赖项的词。要注意两点：
- 只有一个词作为根结点（Root）的依赖项
- 不能构成环(A->B, B->A)

**依存分析的方法** 
1. Dynamic programming：同时也可以做成份文法的一种方法。
2. Graph algorithms(图算法)：通常使用(Minimum Spanning Tree, MST)最小生成树算法来做。
3. Constraint Satisfaction(约束补偿)
4. **Transition-based parsing(基于转换的依存分析)**：也被经常成为deterministic dependency parsing(确定型依存句法分析)：目前做依存分析主流的方法。贪婪地选择依赖项通过机器学习分类器。

#######Arc-standard transition-based parser(基于弧标准转换的依存分析)
下面分析这个例子："I ate fish."
左侧灰色区域是一个栈，而橙色区域是一个buffer缓冲区，里面存放的是我们即将处理的句子。
我们将根据这个转换标准下进行三个动作：
![avatar](https://github.com/coderGray1296/NLP/blob/master/cs224n/pictures/4.4.png)
1. Shift：将缓冲区顶端的单词存入栈顶。
2. Left-Arc：将左侧加上的词，做出附加判断。
3. Right-Arc：将右侧加上的词，作出附加判断。
例如在本例子中：
Left-Arc操作中，I是ate的依赖项，然后把这个依赖项从栈中取出，增加一个弧，这就是一个Left-Arc操作。Right-Arc操作类似。该进行何种操作是通过一个机器学习分类器决定的，常用的特征有栈顶的元素及其对应词性词性，缓冲区中第一个词及其对应词性等等。
![avatar](https://github.com/coderGray1296/NLP/blob/master/cs224n/pictures/4.5.png)
