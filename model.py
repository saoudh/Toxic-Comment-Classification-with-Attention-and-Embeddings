import tensorflow as tf


def spatial_dropout_layer(x, keep_prob, seed=1234):
    # x is a convnet activation with shape BxWxHxF where F is the
    # number of feature maps for that layer
    # keep_prob is the proportion of feature maps we want to keep

    # get the batch size and number of feature maps
    num_feature_maps = [tf.shape(x)[0], tf.shape(x)[3]]

    # get some uniform noise between keep_prob and 1 + keep_prob
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(num_feature_maps,
                                       seed=seed,
                                       dtype=x.dtype)

    # if we take the floor of this, we get a binary matrix where
    # (1-keep_prob)% of the values are 0 and the rest are 1
    binary_tensor = tf.floor(random_tensor)

    # Reshape to multiply our feature maps by this tensor correctly
    binary_tensor = tf.reshape(binary_tensor,
                               [-1, 1, 1, tf.shape(x)[3]])
    # Zero out feature maps where appropriate; scale up to compensate
    ret = tf.div(x, keep_prob) * binary_tensor
    return ret


class Attention:
    def __call__(self, inp, combine=True, return_attention=True):
        # Expects inp to be of size (?, number of words, embedding dimension)
        repeat_size = int(inp.shape[-1])
        # Map through 1 Layer MLP
        # tanh(Wh+b)
        x_a = tf.contrib.layers.fully_connected(
            inp,
            repeat_size,
            activation_fn=tf.nn.tanh,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.zeros_initializer(),
            trainable=True,
            scope="tanh_MLP1"
        )

        # Dot with word-level vector
        # linear layer means no activation-function
        x_a = tf.contrib.layers.fully_connected(
            x_a,
            1,
            activation_fn=None,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.zeros_initializer(),
            trainable=True,
            scope="tanh_MLP2"
        )
        # flatten the input, assuming first dim is batch -> [batch_size,k]
        x_a = tf.contrib.layers.flatten(
            x_a,
            outputs_collections=None,
            scope="flatten"
        )
        # attention=softmax(v*u) =softmax(x_a)
        att_out = tf.nn.softmax(logits=x_a)
        # Clever trick to do elementwise multiplication of alpha_t with the correct h_t:
        # RepeatVector will blow it out to be (?,120, 200)
        # Then, Permute will swap it to (?,200,120) where each row (?,k,120) is a copy of a_t[k]
        # Then, Multiply performs elementwise multiplication to apply the same a_t to each
        # dimension of the respective word vector
        x_a2 = tf.tile(tf.expand_dims(att_out, axis=1), [1, repeat_size, 1])
        # permute to get the transpose
        x_a2 = tf.transpose(x_a2, perm=[0, 2, 1])
        out = tf.multiply(inp, x_a2)
        if combine:
            # Now we sum over the resulting word representations
            out = tf.reduce_sum(out, axis=1, name="expect_over_words")

        if return_attention:
            out = (out, att_out)

        return out

class Model:
    def __init__(self,config):
        self.maxlen=config.maxlen
        self.embed_size=config.embed_size
        self.max_features = config.max_features
        self.mini_batch_size = config.mini_batch_size
        self.lstm_shape=config.lstm_shape
        self.config=config
        # variable initialization
        # 6 classes possible
        self.toxic_labels = tf.placeholder(dtype=tf.float32, shape=[None, 6],name="toxic_labels")
        self.inp = tf.placeholder(shape=(None, self.maxlen), dtype=tf.int32,name="inputs")
        if config.use_glove_embeddings:
            # using placeholder to not copy multiple times in graph. With initial. embed. as constant in Variable not efficient
            self.emb = tf.Variable(tf.constant(0.0, shape=[config.vocab_size, config.embed_size]),
                                   trainable=False, name="Word_embedding_glove")
            self.embedding_placeholder = tf.placeholder(tf.float32, [config.vocab_size, config.embed_size])
            self.embedding_init = self.emb.assign(self.embedding_placeholder)
        else:
            self.rand = tf.random_uniform([self.max_features, self.embed_size], -1, 1, dtype=tf.float32)
            # [batch_size,max_len,embed_size] , embed_size = feature_size, max_len= sequence/time length
            # number of words in the vocabulary = max_features
            self.emb = tf.Variable(self.rand,name="Word_embedding_random", dtype=tf.float32)


        self.optimizer = tf.train.AdamOptimizer()


    def build_model(self):
        # Define the model

        # looks up IDs in a single tensor or a list of tensors in params
        embedded_word_ids = tf.nn.embedding_lookup(params=self.emb, ids=self.inp)
        # because especially in images normal dropout doesnt work well because adjacent pixel correlate,
        # dropout over dimensions is helpful instead of arbitrary and indepenend dropout as an usual dropout
        # x= spatial_dropout(emb,keep_prob=0.35)
        # unstack input into sequence length of words in a sentence -> timesteps=sequence-length, over column 1
        # unstacked_input=tf.unstack(embedded_word_ids,maxlen,1)
        lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(self.lstm_shape, forget_bias=1.0)
        lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(self.lstm_shape, forget_bias=1.0)
        self.outputs, output_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, embedded_word_ids,
                                                                 dtype=tf.float32,scope="bi-directional-RNN")
        self.x=tf.concat(self.outputs, 1)
        if self.config.use_attention:
            # concat. the forward and backward bi-RNN output
            self.x, self.attention = Attention()(self.x)
        x_a = tf.contrib.layers.fully_connected(
            self.x,
            6,
            activation_fn=None,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.zeros_initializer(),
            trainable=True,
            scope="sigmoid_MLP-output"
        )

        # for multilabel binary classif.: sigmoid with cross entropy, as more than 1 label can be true
        # for 1 label binary classif.: softmax with cross entropy
        self.toxic_cross_entr = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_a, labels=self.toxic_labels)
        self.prediction_sigmoid=tf.sigmoid(x_a)
        # for prediction the probability has to be rounded
        self.prediction=tf.round(self.prediction_sigmoid)
        self.correct_prediction = tf.equal(self.prediction, self.toxic_labels)

        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
        self.toxic_classif_loss = tf.reduce_mean(self.toxic_cross_entr)
        self.toxic_classif_optim = self.optimizer.minimize(loss=self.toxic_classif_loss)
    
    def assign_embeddings(self,sess,embeddings):
        sess.run(self.embedding_init, feed_dict={self.embedding_placeholder: embeddings})

    def get_word_importances(self,X_t,sess,emb_word_vocab):
        pred,att = sess.run([self.prediction,self.attention], feed_dict={self.inp: X_t})

        return pred, [(emb_word_vocab.index(word), importance) for word, importance in zip(X_t[0], att[0]) if
                   word in emb_word_vocab]

