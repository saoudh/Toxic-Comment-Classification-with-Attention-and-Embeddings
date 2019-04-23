from __future__ import print_function

import sys
import codecs
import numpy as np

import os

def main():
    '''Convert an embedding file in a textual format (such as the pretrained GloVe embeddings) to a binary format.

    The input is a text file with a '.txt' extension, in which each line is space-separated, the first word being the target word
    and the rest being the textual representation of its vector.
    The output is two files, saved in the same directory as the input file:
    1) a binary file containing the word matrix (to load using np.load(file)), saved with the extension 'npy'
    2) a text file containing the vocabulary (one word per line, in order), saved with the extension 'vocab'

    Usage:
        convert_text_embeddings_to_binary.py <embedding_file>
        <embedding_file> = the input embedding file
    '''

    embedding_file = "/Users/admin/workspace/embeddings/glove.6B/glove.6B.100d.txt"
    out_emb_file, out_vocab_file = embedding_file.replace('.txt', ''), embedding_file.replace('.txt', '.vocab')

    if os.path.exists(out_emb_file and out_vocab_file):
        print("embeddings files seems to be already generated!\nloading files...")
        wv, index2word, word2index=read_embeddings(out_emb_file)
        return wv, index2word, word2index
    print('Loading embeddings file from {}'.format(embedding_file))
    wv, words = load_embeddings(embedding_file)

    print('Saving binary file to {}'.format(out_emb_file))
    np.save(out_emb_file, wv)

    print('Saving vocabulary file to {}'.format(out_vocab_file))
    with codecs.open(out_vocab_file, 'w', 'utf-8') as f_out:
        for word in words:
            f_out.write(word + '\n')
    wv, index2word, word2index = read_embeddings(out_emb_file)
    return wv, index2word, word2index
    #return wv,words


def load_embeddings(file_name):
    """
    Load the pre-trained embeddings from a file
    :param file_name: the embeddings file
    :return: the vocabulary and the word vectors
    """
    with codecs.open(file_name, 'r', 'utf-8') as f_in:
        lines = [line.strip() for line in f_in]

    embedding_dim = len(lines[0].split()) - 1
    words, vectors = zip(*[line.strip().split(' ', 1) for line in lines if len(line.split()) == embedding_dim + 1])
    wv = np.loadtxt(vectors)

    return wv, words


def read_embeddings(embeddings_path):
    print("read_embeddings")
    # if filepath has an extension, then remove it
    embeddings_path = embeddings_path.replace('.txt', '')
    with codecs.open(embeddings_path + '.vocab', 'r', 'utf-8') as f_in:
        index2word = [line.strip() for line in f_in]

    word2index = {w: i for i, w in enumerate(index2word)}
    wv = np.load(embeddings_path + '.npy')
    print("read_embeddings-wv[:3]=",wv[:3])
    print("read_embeddings-index2word.type=",type(index2word))
    print("read_embeddings-word2index.type=",type(word2index))

    return wv, index2word, word2index


if __name__ == '__main__':
    wv, index2word, word2index = main()
    print("wv.len=",len(wv))
    print("index2word.len=",len(index2word))
    print("word2index.len=",len(word2index))

    print("np.shape(wv)=",np.shape(wv))
    print("np.shape(index2word)=",np.shape(index2word))
    print("np.shape(word2index)=",np.shape(word2index))

    print("wv.type=", type(wv))
    print("index2word.type=", type(index2word))
    #print("word2index.type=", type(word2index))
    for k, v in word2index.items():
        print("word: ",k, " - index: ",v)
        if v==10:
            break
    print("wv[:10]=",wv[:10])
    print("index2word[:10]=",index2word[:10])
    print("word2index[:10]=",word2index[:10])

