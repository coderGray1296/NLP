### ELMo (Embeddings from Language Models)
这篇文章作为ELMo论文的补充，针对网上的一些解读让我对于ELMo的理解更加细节。**其实预训练语言模型像word2vec一样会用就行，但是我认为理解这些好的论文模型设计的细节和创新点会开拓我们的视野，在进行我们自己领域的创新研究时候也许会成为思路源泉。**
本质上，之前的Word Embedding是一个静态的表示方式，所谓静态也就是指训练好的每个单词的表示就固定住了，以后使用的时候不会考虑特定场景下不同任务的上下文信息，这也是问题所在。
**这里也不会细究数学表达式，在论文中解读过。**
正如论文中说的，训练好的语言模型对于一个单词来说会获得三个embedding，分别是文章提到的pre-trained的language model用了两层的biLM获得的两个表示，还有对token进行上下文无关编码的表示，**这是通过CNN对字符级进行编码得到的特征**。
ELmo的本质是：事先用一个语言模型训练一个单词的Word Embedding，此时多义词没法区分，不过没关系。在实际使用Word Embedding的时候，单词已经具备了特定的上下文，这个时候再根据上下文去调整单词的Word Embedding表示，再经过了这样的调整后词向量就更能表达上下文的含义，从而解决了多义词的问题。**所以，ELMo本身就是根据当前上下文对Word Embedding动态调整的思路。**
![avatar](https://github.com/coderGray1296/NLP/blob/master/ELMo/pictures/3.jpg)
**ELMo采用了典型的两阶段过程，第一个阶段是利用语言模型进行预训练；第二个阶段是在做下游任务时，从预训练网络中提取对应单词的Word Embedding作为新特征补充道下游任务中。**
使用这个网络结构利用大量语料做语言模型任务就能预先训练好这个网络，如果训练好这个网络后，输入一个新句子Snew，**句子中每个单词都能得到对应的三个Embedding:最底层是单词的 Word Embedding，往上走是第一层双向LSTM中对应单词位置的 Embedding，这层编码单词的句法信息更多一些；再往上走是第二层LSTM中对应单词位置的 Embedding，这层编码单词的语义信息更多一些。**也就是说，ELMO 的预训练过程不仅仅学会单词的 Word Embedding，还学会了一个双层双向的LSTM网络结构，而这两者后面都有用。**很明显ELMo是基于特征的预训练模型**
![avatar](https://github.com/coderGray1296/NLP/blob/master/ELMo/pictures/4.jpg)
![avatar](https://github.com/coderGray1296/NLP/blob/master/ELMo/pictures/5.jpg)
但是ELMo同样存在着不足，LSTM与Transformer比起来提取特征的能力有限，同时，**ELMo采用双向拼接来融合特征的方式，可能会比Bert一体化的融合特征的方式弱一些。**后者并没有验证，只是一种可能性。
