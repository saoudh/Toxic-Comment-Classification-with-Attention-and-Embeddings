import tensorflow as tf

from save_embeddings_to_bin import *


def generate_embeddings(config,vocabulary):
    print("generate_embeddings..")
    glove_filename = "glove.6B.{}d.vocab".format(config.embed_size)
    file_path=os.path.join(config.glove_dir_path,glove_filename)
    print("file path glove=",file_path)
    if os.path.isfile(file_path):
        wv, index2word, word2index = read_embeddings(file_path)
        word_vocab = []

        embedding_matrix = []
        print("index2word.shape=",np.shape(index2word))
        print("word2index.shape=", np.shape(index2word))
        print("vocabulary.shape=",np.shape(vocabulary))

        print("index2word[:10]=",index2word[:10])
        print("word2index=", index2word[:10])
        print("vocabulary[:10]=",vocabulary[:10])
        # word is a tuple of (word,nr-of-occurrence)
        for i,word in enumerate(vocabulary):
            if word in index2word:
                print("word=", word, "word.index=", index2word.index(word))

                if i%50==0:
                    print("i=",i,"-word=",word)
                word_vocab.append(word)
                embedding_matrix.append(wv[index2word.index(word)])
        '''for emb_word,vector in zip(index2word,wv):
            print("emb_word: ",emb_word)
            # if word from glove exists in covered vocabulary
            if emb_word in vocabulary:
                word_vocab.append(emb_word)
                embedding_matrix.append(vector)
        '''
        print("embedding_matrix.shape",np.shape(embedding_matrix))
        print("embedding_matrix[:3]=",np.shape(embedding_matrix))

        return word_vocab,embedding_matrix
    else:
        raise Exception("no glove file at{} available!".format(file_path))


def generate_vocabulary(list_sentences,config):
    print("generate vocabulary...")
    import nltk
    #allWords = nltk.tokenize.word_tokenize(list_sentences)

    vocab = []

    for sentence in list_sentences:
        for word in sentence.split():
            # keep capital letters for entity consideration
            vocab.append(word)
    allWordDist = nltk.FreqDist(vocab)
    mostCommon = allWordDist.most_common(config.max_features)
    # remove distribution number elements in tuple
    mostCommon=[i for i,_ in mostCommon]
    return mostCommon

def generate_indexes(vocabulary):
    print("generate indexes...")
    int_to_vocab = {}
    symbols = {0: 'PAD', 1: 'UNK'}

    for index_no, word in enumerate(vocabulary):
        int_to_vocab[index_no] = word
    int_to_vocab.update(symbols)
    vocab_to_int={word: index_no for index_no, word in int_to_vocab.items()}
    print("int_to_vocab.len",len(int_to_vocab))
    print("int_to_vocab.shape",np.shape(int_to_vocab))
    print("int_to_vocab",int_to_vocab)
    return int_to_vocab,vocab_to_int

def generate_sequences(data,vocab_to_int):
    print("generate sequences...")
    encoded_data = []

    for sentence in data:
        sentence_ = []
        for word in sentence.split():
            if word in vocab_to_int:
                sentence_.append(vocab_to_int[word])
        encoded_data.append(sentence_)
    print("encoded_data.len",len(encoded_data))
    print("encoded_data.shape",np.shape(encoded_data))
    print("encoded_data[:10]",encoded_data[:10])

    return encoded_data

def trim_sequences(data,emb_word_vocab):
    print("trim sequences according to embedding vocabulary considering padding and unknown words...")
    encoded_data = []

    for i,sentence in enumerate(data):
        sentence_ = []
        for word in sentence.split():
            if word in emb_word_vocab:
                index_of_word=emb_word_vocab.index(word)
            else:
                # index for "unknown" words, i.e. words not contained in embeddings vocab., is 1
                index_of_word=1
            sentence_.append(index_of_word)
        encoded_data.append(sentence_)
        if i<10:
            print("sentence: ",sentence)
            print("sequence: ",sentence_,"\n")

    print("encoded_data.len", len(encoded_data))
    print("encoded_data.shape", np.shape(encoded_data))
    print("encoded_data[:10]", encoded_data[:10])

    return encoded_data

def save_model_to_file(sess,config):
    saver = tf.train.Saver()
    # create directory for model-files if it doesn't exist
    if not os.path.isdir("/".join(config.model_file_path.split("/")[:-1])):
        print("directory " + "/".join(config.model_file_path.split("/")[:-1]) + " doesn't exist ... creating directory")
        os.makedirs(os.path.dirname(config.model_file_path), exist_ok=True)
    # save all session variable values to model file
    save_path = saver.save(sess, config.model_file_path)
    print("Model saved in path: %s" % save_path)


def load_model_from_file(sess,config):
    if not os.path.isdir("/".join(config.model_file_path.split("/")[:-1])):
        print("path %s doesn't exist!" %"/".join(config.model_file_path.split("/")[:-1]))
        return
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    saver.restore(sess, config.model_file_path)
    print("Model at %s restored successfully!" %"/".join(config.model_file_path.split("/")[:-1]))
    return sess

def save_np_array_to_bin_file(data_arr, config,filename):
    filename = config.log_dir_path + filename
    # create directory if it doesn't exist
    if not os.path.isdir(config.log_dir_path):
        os.makedirs(config.log_dir_path, exist_ok=True)
    np.save(filename, np.reshape(data_arr, [-1]))

def load_np_array_from_bin_file(data_arr, config,filename):
    filename = config.log_dir_path + filename
    # for multiple experiments runs: as long as the log-filename already exists,
    # increment the ending of the filename
    if os.path.isfile(filename):
        arr= np.load(filename)
        print("array loaded successfully from binary log-file in path %s!"%filename)
        return arr
    else:
        print("log-file in path %s doesn't exist!"%filename)


def reset_values(check_variables, dic_var):
    for k in check_variables:
        dic_var[k] = []

def update_var(dic_var, var_name, var_val):
    dic_var[var_name] = dic_var[var_name] + [var_val]

def print_vars(dic_var):
    for k in dic_var:
        print(k, np.mean(dic_var[k]))