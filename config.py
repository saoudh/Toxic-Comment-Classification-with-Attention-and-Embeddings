import tensorflow as tf

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

config=flags.FLAGS