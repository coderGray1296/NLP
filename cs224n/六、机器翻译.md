# RNN and language model
### 传统语言模型
传统的语言模型通常是基于一个窗口值，在窗口值的基础上来预测下一个位置的概率，即：满足马尔可夫假设。而先验概率是根据共现频率来决定的，因此如果窗口的值越大，需要计算的步骤就成指数型增长，对于内存有很大的压力，因此目前来说**Recurrent Neural Network**成为了主流模型。
### 递归神经网络语言模型
与传统语言模型的不同的是，递归神经网络考虑了当前预测位置之前的所有信息，通过隐状态来传递这个信息，但是又不需要每次都进行计算整体，因此计算量相比传统的语言模型小得多。
![avatar](https://github.com/coderGray1296/NLP/blob/master/cs224n/pictures/5.1.png)
![avatar](https://github.com/coderGray1296/NLP/blob/master/cs224n/pictures/5.2.png)
列向量L是embedding矩阵在t时刻t index
![avatar](https://github.com/coderGray1296/NLP/blob/master/cs224n/pictures/5.3.png)
### Vanishing gradient problem
![avatar](https://github.com/coderGray1296/NLP/blob/master/cs224n/pictures/5.4.png)
![avatar](https://github.com/coderGray1296/NLP/blob/master/cs224n/pictures/5.5.png)
![avatar](https://github.com/coderGray1296/NLP/blob/master/cs224n/pictures/5.6.png)
当W随即初始化一些很小的数值(小于1)时，随着序列长度的增加，雅可比矩阵的梯度会越来越小最终消失，而且这个过程是很快的；反之如果W被初始化的大(大于1)，随着序列长度t的增长，最终会出现梯度爆炸，由于这个计算量难以估计，最终爆炸。
### 双向RNN
![avatar](https://github.com/coderGray1296/NLP/blob/master/cs224n/pictures/5.7.png)
从左到右的递归预测的词只能通过上文参考，双向可以同时从上下文参考信息。
### 评价指标 F1值
![avatar](https://github.com/coderGray1296/NLP/blob/master/cs224n/pictures/5.8.png)
