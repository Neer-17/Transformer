import numpy as np
with open('\\data\\input.txt','r+',encoding='utf-8-sig') as txt_file:
    text = txt_file.read()

tokens = []
for char in text:
    tokens.append(char)

vocab = list(set(tokens))
vocab = sorted(vocab)
vocab_size = len(vocab)
itos = {}
stoi = {}

for index, char in enumerate(vocab):
    itos[index] = char
    stoi[char] = index

def encode(text:str)->list:
    en_list = []
    for char in text:
        en_list.append(stoi[char])
    return en_list

def decode(en_list:list)->str:
    snt = []
    for index in en_list:
        snt.append(itos[index])
    return "".join(snt)

def split_data(encoded_text:list)->list:
    train_data = encoded_text[:int(0.9*(len(encoded_text)))]
    val_data = encoded_text[int(0.9*len(encoded_text)):]
    return train_data, val_data

def get_batch(block_size:int,batch_size:int,train_data:list)->list:
    x = []
    y = []
    for i in range(batch_size):
        st_index = np.random.randint(low=0,high=len(train_data)-(block_size+1))
        x.append(train_data[st_index:st_index+block_size])
        y.append(train_data[st_index+1:st_index+block_size+1])
    return x,y
