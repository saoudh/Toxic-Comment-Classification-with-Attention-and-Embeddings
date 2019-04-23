import numpy as np
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from model import *
from utils import *

def _get_data(train):
    train = train.sample(frac=1)
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    y = train[list_classes].values
    return train, list_classes, y

# useful for char-embeddings: tokenizes into int-characters instead of words of form [my, sentence] -> [[3,4],[5,3,4,6,1,22,55,22]]
def _preprocessing_old(list_sentences,config):
    # Tokenize char in words
    tokenizer = Tokenizer(num_words=config.max_features)
    tokenizer.fit_on_texts(list(list_sentences))
    list_tokenized_train = tokenizer.texts_to_sequences(list_sentences)
    print("list_tokenized_train: ", list_tokenized_train[0], "-type=",type(list_tokenized_train[0]))
    print("tok test=",tokenizer.sequences_to_texts([[0]] ))

    text = tokenizer.sequences_to_texts([list_tokenized_train[0]] )
    print("sequences_to_texts: ", text)

    # Pad
    X_t = pad_sequences(list_tokenized_train, maxlen=config.maxlen)

    return X_t,tokenizer


dic_var = {}
check_variables = ["loss", "accuracy"]
reset_values(check_variables, dic_var)


def _train(X_t,y,model,sess):
    ##### creating model ######
    print("creating model: ")
    # loop over batches for training

    batch_size = 100
    from tqdm import tqdm
    for episode in tqdm(range(int(np.ceil(len(X_t) / batch_size)))):
        batch_start_idx = episode * batch_size
        batch_end_idx = (episode + 1) * batch_size
        batch_end_idx = None if batch_end_idx > len(X_t) else batch_end_idx
        print("i=", episode)
        '''print("batch-start=", batch_start_idx)
        print("batch-end=", batch_end_idx)
        print("model.inp=",model.inp)
        print("model.toxic_labels=",model.toxic_labels)
        print("X_t[batch_start_idx:batch_end_idx]=",X_t[batch_start_idx:batch_end_idx])
        print("y[batch_start_idx:batch_end_idx]=",y[batch_start_idx:batch_end_idx])
        '''
        accuracy, prediction_sigmoid,prediction,correct_pred,toxic_cross_entropy,\
        toxic_labels,pred,_, loss = sess.run([model.accuracy,model.prediction_sigmoid,
                                              model.prediction,

                                                                            model.correct_prediction,model.toxic_cross_entr,model.toxic_labels,model.prediction,model.toxic_classif_optim, model.toxic_classif_loss], feed_dict={model.inp: X_t[batch_start_idx:batch_end_idx],
                                                                                 model.toxic_labels: y[
                                                                                               batch_start_idx:batch_end_idx]})
        update_var(dic_var, "loss", loss)
        update_var(dic_var, "accuracy", accuracy)

        print("loss=", loss)
        '''print("toxic_cross_entropy[:3]=",toxic_cross_entropy[:3])
        print("toxic_labels[:3]=",toxic_labels[:3])
        print("pred=",pred)'''

        #print("correct_pred=",correct_pred)
        print("accuracy=",accuracy)
        #print("pred_sigmoid=",prediction_sigmoid)'''
    # save model
    save_model_to_file(sess, config)
    # save
    for i in check_variables:
        save_np_array_to_bin_file(dic_var[i],config,i)



def _test(X_t,y,model,sess):
    ##### test #######
    print("Test:")
    mytext = "I can\'t make any real suggestions on improvement"
    print("creating model: ")
    # loop over batches for training

    batch_size = 100

    for i in range(int(np.ceil(len(X_t) / batch_size))):
        batch_start_idx = i * batch_size
        batch_end_idx = (i + 1) * batch_size
        batch_end_idx = None if batch_end_idx > len(X_t) else batch_end_idx
        print("i=", i)
        print("batch-start=", batch_start_idx)
        print("batch-end=", batch_end_idx)
        accuracy = sess.run([model.accuracy], feed_dict={model.inp: X_t[batch_start_idx:batch_end_idx],
                                                                                 model.toxic_labels: y[
                                                                                               batch_start_idx:batch_end_idx]})
        print("accuracy=", accuracy)


def _get_word_importance(X_t,model,sess,emb_vocabulary):
    ##### test #######
    print("_get_word_importance:")
    mytext = "I can\'t make any real suggestions on improvement"
    pred, importance = model.get_word_importances(X_t, model, sess, emb_vocabulary)
    print("prediction: ", pred, " - importance: ", importance)



def _run(config):
    print("preprocessing: ")

    # get list of sentences from file
    data, list_classes, y = _get_data(pd.read_csv(config.training_data_path))
    list_sentences = data['comment_text'].values
    vocabulary=generate_vocabulary(list_sentences,config)
    emb_word_vocab=None
    if config.use_glove_embeddings:
        emb_word_vocab, embeddings = generate_embeddings(config, vocabulary)
        # insert padding at index 0 to vocabulary and as a zero vector to embeddings
        emb_word_vocab.insert(0,'PAD')
        vector= np.zeros((1,100))[0]
        embeddings.insert(0,vector)
        # insert unknown at index 1 to vocabulary and as a random vector to embeddings
        emb_word_vocab.insert(1, 'UNK')
        vector= np.random.uniform(-1.0, 1.0, (1,100))[0]
        embeddings.insert(1, vector)
        # set vocab size of embeddings
        print("np.shape(embeddings)[0]=",np.shape(embeddings)[0])
        config.vocab_size=np.shape(embeddings)[0]

    else:
        vocabulary.insert(0, 'PAD')
        vocabulary.insert(1, 'UNK')
        embeddings = None
    #int_to_vocab, vocab_to_int=generate_indexes(vocabulary)
    # if embedding is not used, then use vocabulary
    if emb_word_vocab is not None:
        trimmed_sequences=trim_sequences(list_sentences,emb_word_vocab)
    else:
        trimmed_sequences=trim_sequences(list_sentences,vocabulary)

    # padding the generated sequences
    padded_sequences=pad_sequences(trimmed_sequences,maxlen=config.maxlen)
    print("padded sequences: ",padded_sequences[:5])

    model=Model(config)
    model.build_model()

    sess = tf.Session()
    init = tf.global_variables_initializer()

    # for storing the model to file
    saver = tf.train.Saver()
    if config.load_model:
        sess = load_model_from_file(sess, config.model_file_path)
    else:
        sess.run(init)
    model.assign_embeddings(sess,embeddings)
    print("1 X_t.type=", type(padded_sequences), " - y.type=", type(y))
    print("1 X_t.shape=", np.shape(padded_sequences), " - y.shape=", np.shape(y))
    print("1 X_t[0][:10]=", padded_sequences[:10][:10], " - y[0]=", y[:10])

    if config.mode=="test":
        _test(padded_sequences,y,model,sess)
    if config.mode=="train":
        _train(padded_sequences,y,model,sess)
    if config.mode == "word_importance":
        # todo: pass test-sequences instead training ones
        _get_word_importance(padded_sequences, model, sess,emb_word_vocab)

flags = tf.app.flags
# model parameters
flags.DEFINE_integer("max_features", 30000, "number of words in vocabulary")
flags.DEFINE_integer("embed_size", 100, "embedding size of a single token")
flags.DEFINE_integer("mini_batch_size", 40, "mini batch size")
flags.DEFINE_integer("vocab_size", 1, "size of vocab, will be overwritten in code after generating vocab.")
flags.DEFINE_integer("maxlen", 200, "max number of words/tokens in sentence")
flags.DEFINE_integer("lstm_shape", 60, "hidden size of LSTM-Cell")

flags.DEFINE_boolean("use_glove_embeddings", True, "whether using glove embeddings [True|False]")
flags.DEFINE_boolean("use_attention", True, "whether using attention [True|False]")
flags.DEFINE_boolean("load_model", False, "loading model from file [True|False]")

# paths
flags.DEFINE_string("training_data_path", 'data/train.csv', "training data path as csv-file")
flags.DEFINE_string("model_file_path", 'model_files/mymodel', "model file path")
flags.DEFINE_string("log_dir_path", 'logs/', "path to log directory path")
flags.DEFINE_string("glove_dir_path", '../embeddings/glove.6B/', "embeddings dir path as zip")
# training parameters
flags.DEFINE_string("mode", "train", "mode: [train | test]")


if __name__ == "__main__":
    config = flags.FLAGS
    _run(config)