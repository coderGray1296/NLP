import pickle

buf = open('./data_path/word2id.pkl','rb')
word = pickle.load(buf, encoding='utf-8')
print(word)