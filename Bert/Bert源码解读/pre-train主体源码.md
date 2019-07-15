## 本篇文章主要解读预训练的代码pre-train
pre-train是迁移学习的基础，虽然Google已经发布了各种预训练好的模型，而且因为资源消耗巨大，自己再预训练也不现实（在Google Cloud TPU v2 上训练BERT-Base要花费近500刀，耗时达到两周。在GPU上可想而知只会更贵），但是学习bert的预训练方法可以为我们弄懂整个bert的运行流程提供莫大的帮助。
**pre-train涉及到的模块为以下三个：**
- tokenization.py
- create_pretraining_data.py
- run_pretraining.py

其中tokenization是对原始句子内容的解析，分为**BasicTokenizer**和**WordpieceTokenizer**两个，不只是在预训练中，在fine-tune和推断过程同样要用到它；create_pretraining_data顾名思义就是**将原始语料转换成适合模型预训练的输入数据**；run_pretraining就是预训练的执行代码了。
### tokenization.py
##### BasicTokenizer
```
class BasicTokenizer(object):
  """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

  def __init__(self, do_lower_case=True):
    self.do_lower_case = do_lower_case

  def tokenize(self, text):
    """Tokenizes a piece of text."""
    text = convert_to_unicode(text)
    text = self._clean_text(text)

    text = self._tokenize_chinese_chars(text)

    orig_tokens = whitespace_tokenize(text)
    split_tokens = []
    for token in orig_tokens:
      if self.do_lower_case:
        token = token.lower()
        token = self._run_strip_accents(token)
      split_tokens.extend(self._run_split_on_punc(token))

    output_tokens = whitespace_tokenize(" ".join(split_tokens))
    return output_tokens

  def _run_strip_accents(self, text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
      cat = unicodedata.category(char)
      if cat == "Mn":
        continue
      output.append(char)
    return "".join(output)

  def _run_split_on_punc(self, text):
    """Splits punctuation on a piece of text."""
    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
      char = chars[i]
      if _is_punctuation(char):
        output.append([char])
        start_new_word = True
      else:
        if start_new_word:
          output.append([])
        start_new_word = False
        output[-1].append(char)
      i += 1

    return ["".join(x) for x in output]

  def _tokenize_chinese_chars(self, text):
    """Adds whitespace around any CJK character."""
    output = []
    for char in text:
      cp = ord(char)
      if self._is_chinese_char(cp):
        output.append(" ")
        output.append(char)
        output.append(" ")
      else:
        output.append(char)
    return "".join(output)

  def _is_chinese_char(self, cp):
    """Checks whether CP is the codepoint of a CJK character."""
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
        (cp >= 0x3400 and cp <= 0x4DBF) or  #
        (cp >= 0x20000 and cp <= 0x2A6DF) or  #
        (cp >= 0x2A700 and cp <= 0x2B73F) or  #
        (cp >= 0x2B740 and cp <= 0x2B81F) or  #
        (cp >= 0x2B820 and cp <= 0x2CEAF) or
        (cp >= 0xF900 and cp <= 0xFAFF) or  #
        (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
      return True
    return False

  def _clean_text(self, text):
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []
    for char in text:
      cp = ord(char)
      if cp == 0 or cp == 0xfffd or _is_control(char):
        continue
      if _is_whitespace(char):
        output.append(" ")
      else:
        output.append(char)
    return "".join(output)
```
**BasicTokenizer**主要进行的是unicode转换、标点符号分割、小写转换、中文字符分割、去除重音符号等操作，最后返回的是关于词的数组(中文是字的数组)。
##### WordpieceTokenizer
```
class WordpieceTokenizer(object):
  """Runs WordPiece tokenziation."""

  def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
    self.vocab = vocab
    self.unk_token = unk_token
    self.max_input_chars_per_word = max_input_chars_per_word

  def tokenize(self, text):
    text = convert_to_unicode(text)
    output_tokens = []
    for token in whitespace_tokenize(text):
      chars = list(token)
      if len(chars) > self.max_input_chars_per_word:
        output_tokens.append(self.unk_token)
        continue
      is_bad = False
      start = 0
      sub_tokens = []
      while start < len(chars):
        end = len(chars)
        cur_substr = None
        while start < end:
          substr = "".join(chars[start:end])
          if start > 0:
            substr = "##" + substr
          if substr in self.vocab:
            cur_substr = substr
            break
          end -= 1
        if cur_substr is None:
          is_bad = True
          break
        sub_tokens.append(cur_substr)
        start = end

      if is_bad:
        output_tokens.append(self.unk_token)
      else:
        output_tokens.extend(sub_tokens)
    return output_tokens
```
**WordpieceTokenizer**的目的是将合成词分解成类似词根一样的词片。例如将"unwanted"分解成["un", "##want", "##ed"]这么做的目的是防止因为词的过于生僻没有被收录进词典最后只能以[UNK]代替的局面，因为英语当中这样的合成词非常多，词典不可能全部收录。
##### FullTokenizer
```
class FullTokenizer(object):
  """Runs end-to-end tokenziation."""

  def __init__(self, vocab_file, do_lower_case=True):
    self.vocab = load_vocab(vocab_file)
    self.inv_vocab = {v: k for k, v in self.vocab.items()}
    self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
    self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

  def tokenize(self, text):
    split_tokens = []
    for token in self.basic_tokenizer.tokenize(text):
      for sub_token in self.wordpiece_tokenizer.tokenize(token):
        split_tokens.append(sub_token)

    return split_tokens

  def convert_tokens_to_ids(self, tokens):
    return convert_by_vocab(self.vocab, tokens)

  def convert_ids_to_tokens(self, ids):
    return convert_by_vocab(self.inv_vocab, ids)
```
**FullTokenizer**的作用就很显而易见了，对一个文本段进行以上两种解析，最后返回词（字）的数组，同时还提供token到id的索引以及id到token的索引。这里的token可以理解为文本段处理过后的最小单元。
### create_pretraining_data.py
##### 配置
```
flags.DEFINE_string("input_file", None,
                    "Input raw text file (or comma-separated list of files).")
flags.DEFINE_string(
    "output_file", None,
    "Output TF example file (or comma-separated list of files).")
flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")
flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")
flags.DEFINE_integer("max_predictions_per_seq", 20,
                     "Maximum number of masked LM predictions per sequence.")
flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")
flags.DEFINE_integer(
    "dupe_factor", 10,
    "Number of times to duplicate the input data (with different masks).")
flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")
flags.DEFINE_float(
    "short_seq_prob", 0.1,
    "Probability of creating sequences which are shorter than the "
    "maximum length.")
```
- 配置input_file、output_file分别代表输入的源语料文件和处理过的预料文件地址；
- do_lower_case：是否全部转为小写字母，是否转换成小写字母的意义防止将Hello, hello， HELLO等同类词识别为不同的词，增加词典的容量。
- dupe_factor：默认重复10次，目的是可以生成不同情况的masks；
- short_seq_prob：构造长度小于指定**max_seq_length**的样本比例。因为在fine-tune过程里面输入的target_seq_length是可变的（小于等于max_seq_length），那么为了防止过拟合也需要在pre-train的过程当中构造一些短的样本。
##### main入口
```
def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.gfile.Glob(input_pattern))

  tf.logging.info("*** Reading from input files ***")
  for input_file in input_files:
    tf.logging.info("  %s", input_file)

  rng = random.Random(FLAGS.random_seed)
  instances = create_training_instances(
      input_files, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor,
      FLAGS.short_seq_prob, FLAGS.masked_lm_prob, FLAGS.max_predictions_per_seq,
      rng)

  output_files = FLAGS.output_file.split(",")
  tf.logging.info("*** Writing to output files ***")
  for output_file in output_files:
    tf.logging.info("  %s", output_file)

  write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length,FLAGS.max_predictions_per_seq,output_files)
```
从入口看步骤很简单，构造tokenizer--构造instances--保存instances
##### 构造instances
```
def create_training_instances(input_files, tokenizer, max_seq_length,
                              dupe_factor, short_seq_prob, masked_lm_prob,
                              max_predictions_per_seq, rng):
  """Create `TrainingInstance`s from raw text."""
  all_documents = [[]]
  for input_file in input_files:
    with tf.gfile.GFile(input_file, "r") as reader:
      while True:
        line = tokenization.convert_to_unicode(reader.readline())
        if not line:
          break
        line = line.strip()

        # Empty lines are used as document delimiters
        if not line:
          all_documents.append([])
        tokens = tokenizer.tokenize(line)
        if tokens:
          all_documents[-1].append(tokens)

  # Remove empty documents
  all_documents = [x for x in all_documents if x]
  rng.shuffle(all_documents)

  vocab_words = list(tokenizer.vocab.keys())
  instances = []
  for _ in range(dupe_factor):
    for document_index in range(len(all_documents)):
      instances.extend(
          create_instances_from_document(
              all_documents, document_index, max_seq_length, short_seq_prob,
              masked_lm_prob, max_predictions_per_seq, vocab_words, rng))

  rng.shuffle(instances)
  return instances
```
这一步是阅读数据，数据的输入文本可以是一个文件也可以是用逗号分割的若干文件；
文件里用换行来表示句子的边界，即一句一行，同理段落之间用空一行来表示段落的边界，一个段落表示成一个document；具体的构造方法在**create_instances_from_document**函数里面，代码较长，只粘出**create_instances_from_document**的核心部分
```
instance = TrainingInstance(
            tokens=tokens,
            segment_ids=segment_ids,
            is_random_next=is_random_next,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels)
```
1. 一个instance包含一个tlkens，实际上就是一个输入的词序列，该序列的表现形式为：
```
[CLS]A[SEP]B[SEP]
A=[token_0,token-1, ...,token_i]
B=[token_i+1,token_i+2, ...,token_n-1]
其中:
2<= n < max_seq_length - 3 (in short_seq_prob)
n=max_seq_length - 3 (in 1-short_seq_prob)
```
token最后的表现形式如下面所示：
```
Input = [CLS] the man went to [MASK] store [SEP] he bought a gallon [MASK] milk [SEP]
Label = IsNext

Input = [CLS] the man [MASK] to the store [SEP] penguin [MASK] are flight **less birds [SEP]
Label = NotNext
```
- segment_ids指的形式为[0,0,0...1,1,1,1]其中0的个数为i+1歌，1的个数为max_seq_length-(i+1)，对应到模型输入就是token_type
- is_random_next：其实就是上面的Label，0.5的概率为True(和当只有一个segment的时候)，如果为True则B和A不属于同一个document。剩下的情况为False，则B为A同一个document的后续句子。
- masked_lm_positions；序列里被[MASK]的位置
- masked_lm_labels：序列里被[MASK]的token
2. 在create_masked_lm_predictions函数里，一个序列在指定MASK数量后，有80%被真正MASK，10%还是保留原来的token，10%被随机替换成其他的token。

##### 保存instance
```
def write_instance_to_example_files(instances, tokenizer, max_seq_length,
                                    max_predictions_per_seq, output_files):
  """Create TF example files from `TrainingInstance`s."""
  writers = []
  for output_file in output_files:
    writers.append(tf.python_io.TFRecordWriter(output_file))

  writer_index = 0

  total_written = 0
  for (inst_index, instance) in enumerate(instances):
    input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = list(instance.segment_ids)
    assert len(input_ids) <= max_seq_length

    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    masked_lm_positions = list(instance.masked_lm_positions)
    masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
    masked_lm_weights = [1.0] * len(masked_lm_ids)

    while len(masked_lm_positions) < max_predictions_per_seq:
      masked_lm_positions.append(0)
      masked_lm_ids.append(0)
      masked_lm_weights.append(0.0)

    next_sentence_label = 1 if instance.is_random_next else 0

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)
    features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
    features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
    features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
    features["next_sentence_labels"] = create_int_feature([next_sentence_label])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))

    writers[writer_index].write(tf_example.SerializeToString())
    writer_index = (writer_index + 1) % len(writers)

    total_written += 1

    if inst_index < 20:
      tf.logging.info("*** Example ***")
      tf.logging.info("tokens: %s" % " ".join(
          [tokenization.printable_text(x) for x in instance.tokens]))

      for feature_name in features.keys():
        feature = features[feature_name]
        values = []
        if feature.int64_list.value:
          values = feature.int64_list.value
        elif feature.float_list.value:
          values = feature.float_list.value
        tf.logging.info(
            "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

  for writer in writers:
    writer.close()

  tf.logging.info("Wrote %d total instances", total_written)
```
instance保存没什么好说的，只有两点需要注意：
```
while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)
```
1. 之前有short_seq_prob的概率导致样本的长度小于max_predictions_per_seq，这里将这些样本补齐，padding为0，同样还有input_mask和segment_ids；
2. 将instance的is_random_next转化为变量next_sentence_label保存。

### run_pretraining.py
这部分就是预训练的执行模块，里面都是tf常规的训练代码，而且前两部分都已经指导预训练的整个逻辑了，这里再简单介绍一下
##### X和Y的确定
```
input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    masked_lm_positions = features["masked_lm_positions"]
    masked_lm_ids = features["masked_lm_ids"]
    masked_lm_weights = features["masked_lm_weights"]
    next_sentence_labels = features["next_sentence_labels"]
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)
```
**其中input_ids,input_mask,segment_ids作为X，剩下的masked_lm_positions,masked_lm_ids,masked_lm_weights,next_sentence_labels共同作为Y**
##### loss
```
(masked_lm_loss,
     masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
         bert_config, model.get_sequence_output(), model.get_embedding_table(),
         masked_lm_positions, masked_lm_ids, masked_lm_weights)

    (next_sentence_loss, next_sentence_example_loss,
     next_sentence_log_probs) = get_next_sentence_output(
         bert_config, model.get_pooled_output(), next_sentence_labels)

    total_loss = masked_lm_loss + next_sentence_loss
```
可以看到loss 分别由masked_lm_loss和next_sentence_loss组成，masked_lm_loss针对的是语言模型对MASK起来的标签的预测，即上下文语境预测当前词；而next_sentence_loss是对于句子关系的预测。前者在迁移学习中可以用于标注类任务（分词、NER等），后者可以用于句子关系任务（QA、自然语言推理等）。
### 总结
1. tokenization模块：我把它叫做对原始文本段的解析，只有解析过后才能标准化输入；
2. create_pretraining_data模块：对原始数据进行转换，原始数据本是无标签的数据，通过句子的拼接可以产生句子关系的标签，通过MASK可以产生标注的标签，其本质是语言模型的应用；
3. run_pretraining模块：在执行预训练的时候针对以上两种标签分别利用bert模型的不同输出部件，计算loss，然后进行梯度下降优化。



