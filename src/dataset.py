import numpy as np
#READ THE DATA
with open('Transformer\\data\\input.txt','r+',encoding='utf-8-sig') as txt_file:
    text = txt_file.read()

#TOKEN LIST
tokens = list(text)

#VOCABULARY
vocab = list(set(tokens))
vocab = sorted(vocab)
vocab_size = len(vocab)
itos = {}
stoi = {}

#MAPPING
for index, char in enumerate(vocab):
    itos[index] = char
    stoi[char] = index

#ENCODING AND DECODING
def encode(text:str)->list:
    """
    Encodes a string into a list of integers.
    Args:
        text (str): The input string to be encoded.
    Returns:
        list: A list of integers representing the encoded string.
    """
    en_list = []
    for char in text:
        en_list.append(stoi[char])
    return en_list

def decode(en_list:list)->str:
    """
    Decodes a list of integers into a string.
    Args:
        en_list (list): The list of integers to be decoded.
    Returns:
        str: The decoded string.
    """
    snt = []
    for index in en_list:
        snt.append(itos[index])
    return "".join(snt)

#GET TRAINING AND VALIDATION DATA
def split_data(encoded_text:list)->tuple:
    """
    Splits the encoded text into training and validation data.
    Args:
        encoded_text (list): The list of integers representing the encoded text.
    Returns:
        tuple: A tuple containing the training data and validation data.
    """
    train_data = encoded_text[:int(0.9*(len(encoded_text)))]
    val_data = encoded_text[int(0.9*len(encoded_text)):]
    return train_data, val_data

#GET BATCH
def get_batch(block_size:int,batch_size:int,train_data:list)->tuple:
    """
    Gets a batch of training data.
    Args:
        block_size (int): The size of each sequence in the batch.
        batch_size (int): The number of sequences in the batch.
        train_data (list): The list of integers representing the training data.
    Returns:
        tuple: A tuple containing the input sequences and target sequences.
    """
    x = []
    y = []
    for i in range(batch_size):
        st_index = np.random.randint(low=0,high=len(train_data)-(block_size+1))
        x.append(train_data[st_index:st_index+block_size])
        y.append(train_data[st_index+1:st_index+block_size+1])
    return x,y
