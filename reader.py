'''
Created on 1 aug. 2018

@author: Dragos2811
'''
import re
#print(re.split(r'(?![/.])\W', "Mr.Jones says This is@a&test example_cool man+right more/fun 43.35"))
def read_data(fname):
    with open(fname,'r') as f:
        return [x for x in re.split('((?![/.\-"])\W)',f.read()) if x]
#print(len(list(set(read_data("gen_html/1.html")))))
def build_dataset(x,y):
    all_words = []
    all_data_as_array = []
    for i in range (x,y):
        data_as_array = read_data("gen_html/%s.html" %  (i))
        for word in data_as_array:
            all_data_as_array.append(word)
        unique_words = list(set(data_as_array))
        for word in unique_words:
            all_words.append(word)
    all_words = list(set(all_words))
    #print (all_words)
    #print (len(all_words))
    return all_words,all_data_as_array
def create(x,y):
    dictionary,data= build_dataset(x,y)
    char_to_ix = { ch:i for i,ch in enumerate(dictionary) }
    ix_to_char = { i:ch for i,ch in enumerate(dictionary) }
    vocab_size = len(dictionary)
    return char_to_ix,ix_to_char,vocab_size,data