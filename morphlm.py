#!/usr/bin/env python3

import codecs
from distutils.dir_util import mkpath
import math
import time

import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

def my_open(filename, mode='r'):
    if mode == 'wb':
        return open(filename, 'wb')
    else:
        return codecs.open(filename, mode, "utf-8")

def print_info():
    print("===MorphLM info===")
    print("Experiment name:", FLAGS.experiment)
    print("Tensorflow version:", tf.__version__)
    print("Training data:", FLAGS.data_train)
    print("Development data:", FLAGS.data_dev)
    print("N-gram order:", FLAGS.order)
    print("Use morphology:", FLAGS.morph)
    print("Use NCE:", FLAGS.nce)
    if FLAGS.adam:
        print("Optimizer:", "Adam")
    else:
        print("Optimizer:", "SGD")
    print("Maximum iterations:", FLAGS.max_iter)
    print("Batch size:", FLAGS.batch_size)
    print("Embeddings size:", FLAGS.emb_size)
    print("Hidden layer 1 size:", FLAGS.h1_size)
    print("Hidden layer 2 size:", FLAGS.h2_size)
    print("Noise samples:", FLAGS.noise)
    print("Learning rate:", FLAGS.learning_rate)
    print("L1 regularization:", FLAGS.l1_c)
    print("Dropout:", FLAGS.dropout)
    print("Dropout on input:", FLAGS.dropout_emb)
    print("Full tags:", FLAGS.full_tags)
    print("==================")

def define_args():
    flags.DEFINE_string('data-train', None, 'Training data.')
    flags.DEFINE_string('data-dev', None, 'Development data.')
    flags.DEFINE_string('vocab-forms', None, 'Vocabulary for forms.')
    flags.DEFINE_string('vocab-lemmata', None, 'Vocabulary for lemmata.')
    flags.DEFINE_integer('order', 5, 'N-gram order.')
    flags.DEFINE_integer('voc-size', 50000, 'Vocabulary size.')
    flags.DEFINE_integer('emb-size', 200, 'Embedding size.')
    flags.DEFINE_integer('h1-size', 750, 'Embedding size.')
    flags.DEFINE_integer('h2-size', 150, 'Embedding size.')
    flags.DEFINE_integer('max-iter', 5, 'Number of iterations.')
    flags.DEFINE_integer('batch-size', 1000, 'Batch size.')
    flags.DEFINE_integer('keep-models', 3, 'Number of last models to keep.')
    flags.DEFINE_boolean('nce', False, 'Use NCE.')
    flags.DEFINE_boolean('sampled-softmax', True, 'Use sampled softmax.')
    flags.DEFINE_boolean('adam', False, 'Use Adam instead of SGD.')
    flags.DEFINE_boolean('morph', True, 'Use morphology.')
    flags.DEFINE_integer('noise', 10, 'Number of noise samples.')
    flags.DEFINE_string('prefix', './', 'Path prefix.')
    flags.DEFINE_string('experiment', 'test', 'Experiment name.')
    flags.DEFINE_float('l1-c', 0., 'L1 regularization coefficient.')
    flags.DEFINE_float('l2-c', 0., 'L2 regularization coefficient.')
    flags.DEFINE_float('dropout', 1.0, 'Dropout keep rate.')
    flags.DEFINE_boolean('dropout-emb', False, 'Use dropout on embeddings.')
    flags.DEFINE_float('learning-rate', 1., 'Learning rate')
    flags.DEFINE_boolean('full-tags', False, 'Use longer binary reprezentation for tags')

START_WORD = "<s>|<s>|---------------"
END_WORD = "</s>|</s>|Z#-------------"

def ngram_gen(data_file):
    for line in data_file:
        words = (FLAGS.order - 1) * [START_WORD] + line.split() + [END_WORD]
        i = FLAGS.order
        o = FLAGS.order
        l = len(words)
        while i < l:
            yield words[i - o:i]
            i += 1

def make_vec(options, o, lng):
    l = lng * [0]
    if o in options:
        l[options.index(o)] = 1
    return l

def make_tag(tag):
      return sum([make_vec(*x) for x in [("ACDIJNPVRTXZ", tag[0], 12),
      ("FHIMNQTXYZ", tag[2], 10),
      ("DPSWX", tag[3], 5),
      ("1234567X", tag[4], 8),
      ("123X", tag[7], 4),
      ("FHPRX", tag[8], 5),
      ("123", tag[9], 3),
      ("AN", tag[10], 2),
      ("AP", tag[11], 2),
      ("123456789", tag[14], 9)]], [])

def make_full_tag(tag):
      return sum([make_vec(*x) for x in [("ACDIJNPVRTXZ", tag[0], 12),
      ("#%*,}:=?@^456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwyz",
       tag[1], 67),
      ("FHIMNQTXYZ", tag[2], 10),
      ("DPSWX", tag[3], 5),
      ("1234567X", tag[4], 8),
      ("1234567X", tag[5], 8),
      ("FMXZ", tag[6], 4),
      ("123X", tag[7], 4),
      ("FHPRX", tag[8], 5),
      ("123", tag[9], 3),
      ("AN", tag[10], 2),
      ("AP", tag[11], 2),
      ("123456789", tag[14], 9)]], [])

def raw_batch_reader(data_file, batch_size, lmt):
    ngrams = ngram_gen(data_file)
    c = 0
    data_x = []
    data_x2 = []
    data_y = []
    
    vocab_f = {x.strip() : i for i, x in enumerate(my_open(FLAGS.vocab_forms).readlines())}
    vocab_l = {x.strip() : i for i, x in enumerate(my_open(FLAGS.vocab_lemmata).readlines())}

    if FLAGS.full_tags:
        mk_tag = make_full_tag
    else:
        mk_tag = make_tag

    for n in ngrams:
        factors = [x.split('|') for x in n]
        forms = [vocab_f.get(x[0], 0) for x in factors]
        data_y.append(forms)
        lemmata = [vocab_l.get(x[1], 0) for x in factors]
        tags = sum([mk_tag(x[2]) for x in factors], [])
        data_x.append(lemmata)
        data_x2.append(tags)
        c += 1
        if c == batch_size:
            if lmt:
                yield (np.array(data_x), np.array(data_x2, dtype=np.float32),
                       np.array(data_y)), c
            else:
                yield np.array(data_y), c
            data_y = []
            data_x = []
            data_x2 = []
            c = 0

    if c > 0:
        if lmt:
            yield (np.array(data_x), np.array(data_x2, dtype=np.float32), np.array(data_y)), c
        else:
            yield np.array(data_y), c

def create_graph():
    """
    Things to test:
    - different layer sizes
    - different weigth initialization
    """
    h1_size = FLAGS.h1_size
    h2_size = FLAGS.h2_size
    if FLAGS.full_tags:
        morph_one_size = 139
    else:
        morph_one_size = 60
    
    morph_size = morph_one_size * (FLAGS.order - 1)

    def train_sum(name, op, stype=tf.summary.scalar):
        tf.add_to_collection("train_sums", stype(name, op))

    with tf.name_scope("inputs"):
        input_op = tf.placeholder(tf.int32, shape=(None, FLAGS.order - 1))
        if FLAGS.morph:
            input_morph = tf.placeholder(tf.float32, shape=(None, morph_size))
        else:
            input_morph = None
        if FLAGS.morph_out:
            out_morph = tf.placeholder(tf.float32, shape=(None, morph_one_size))


    with tf.name_scope("embeddings_layer"):
        embeddings = tf.Variable(
            tf.random_uniform([FLAGS.voc_size, FLAGS.emb_size], -0.1, 0.1), name='embeddings')
        em_layer_size = (FLAGS.order - 1) * FLAGS.emb_size

        # This does not work with momentum (and AdaGrad) in TF <= 0.6.0:
        embed = tf.reshape(
            tf.nn.embedding_lookup(embeddings, input_op),
            [-1, em_layer_size])
        if FLAGS.dropout_emb:
            dropout_keep_emb = tf.placeholder(tf.float32)
            embed = tf.nn.dropout(embed, dropout_keep_emb)
        if FLAGS.morph:
            embed = tf.concat([embed, input_morph], 1)
            em_layer_size += morph_size

    if FLAGS.dropout < 1.:
        dropout_keep = tf.placeholder(tf.float32)

    def make_hidden(in_l, in_size, size, dropout=False, reg=False):
        with tf.name_scope("hidden_layer"):
            hidden_B = tf.Variable(tf.fill([size], 0.1), name='biases')
            hidden_W = tf.Variable(
                tf.truncated_normal([in_size, size], stddev=0.04),
                name='weights')
            if FLAGS.l2_c > 0. and reg:
                tf.add_to_collection("regularizers", tf.nn.l2_loss(hidden_W))
            relu = tf.nn.relu_layer(in_l, hidden_W, hidden_B)
            if FLAGS.dropout < 1. and dropout:
                return tf.nn.dropout(relu, dropout_keep)
            else:
                return relu

    hidden1 = make_hidden(embed, em_layer_size, h1_size, False, True)
    hidden2 = make_hidden(hidden1, h1_size, h2_size, True)

    objectives = []
    if FLAGS.morph_out:
        with tf.name_scope("morph_out"):
            morph_out_B = tf.Variable(tf.fill([morph_one_size], 0.), name='biases')
            morph_out_W = tf.Variable(tf.random_uniform([h2_size, morph_one_size], -0.5, 0.5), name='weights')
            morph_out_logits = tf.matmul(hidden2, morph_out_W) + morph_out_B
            morph_out_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(morph_out_logits, out_morph))
            objectives.append(morph_out_loss)
            train_sum("loss_morph_out", morph_out_loss)

    with tf.name_scope("softmax_layer"):
        softmax_B = tf.Variable(tf.fill([FLAGS.voc_size], - math.log(FLAGS.voc_size)), name='biases')
        if FLAGS.nce or FLAGS.sampled_softmax:
            nce_W = tf.Variable(
                tf.truncated_normal([FLAGS.voc_size, h2_size],
                                    stddev=1.0 / math.sqrt(FLAGS.emb_size)))
            last_W = nce_W
            out_test = tf.matmul(hidden2, nce_W, transpose_b=True) + softmax_B
        else:
            softmax_W = tf.Variable(
                tf.random_uniform([h2_size, FLAGS.voc_size], -1.0, 1.0))
            last_W = softmax_W
            out_test = tf.matmul(hidden2, softmax_W) + softmax_B
        hist_out = tf.summary.histogram('logits', out_test)
        out_shape = tf.shape(out_test)
        tf.nn.log_softmax(out_test)

    with tf.name_scope("targets"):
        target_op = tf.placeholder(tf.int64, shape=(None))
        target = tf.expand_dims(target_op, 1)
        batch_size = tf.size(target)

    if FLAGS.nce:
        with tf.name_scope("NCE_loss"):
            sampler = tf.nn.learned_unigram_candidate_sampler(target, 1,
                                                              FLAGS.noise, False,
                                                              FLAGS.voc_size)
            loss = tf.reduce_mean(
                tf.nn.nce_loss(nce_W, softmax_B, target, hidden2, FLAGS.noise,
                                           FLAGS.voc_size, sampled_values=sampler))
        with tf.name_scope("cross-entropy_loss"):
            test_loss = tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_op,
                                                        logits=out_test))

    
    elif FLAGS.sampled_softmax:
        with tf.name_scope("SampledSoftmax_loss"):
            sampler = tf.nn.learned_unigram_candidate_sampler(target, 1,
                                                              FLAGS.noise, False,
                                                              FLAGS.voc_size)
            loss = tf.reduce_mean(
                tf.nn.sampled_softmax_loss(nce_W, softmax_B, target, hidden2, FLAGS.noise,
                                           FLAGS.voc_size,
                                           sampled_values=sampler,
                                           partition_strategy='div'))
        with tf.name_scope("cross-entropy_loss"):
            test_loss = tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_op,
                                                        logits=out_test))
    else:
        with tf.name_scope("cross-entropy_loss"):
            crossent = \
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_op,
                                                               logits=out_test)
            loss = tf.reduce_mean(crossent)
            test_loss = loss

    with tf.name_scope("accuracy"):
        accuracy = tf.reduce_sum(tf.cast(tf.nn.in_top_k(out_test, target_op, 1), "float"))
        top_k = tf.reduce_sum(tf.cast(tf.nn.in_top_k(out_test, target_op, 10), "float"))

    train_sum('loss_train', loss)

    objectives.append(loss)

    if FLAGS.l2_c > 0:
        regularizer = FLAGS.l2_c * sum(tf.get_collection("regularizers"))
        train_sum('loss_reg', regularizer)
        objectives.append(regularizer)

    objective = sum(objectives)
    train_sum('loss_objective', objective)

    with tf.name_scope("gradients"):
        if FLAGS.adam:
            emb_opt = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
            grads = emb_opt.compute_gradients(objective)
            hist_grad = tf.summary.histogram('gradient_emb', grads[0][0].values)
            emb_optimizer = emb_opt.apply_gradients(grads[:1])
            rest_optimizer = tf.train.AdamOptimizer().apply_gradients(grads[1:])
            optimizer = tf.group(emb_optimizer, rest_optimizer)
        else:
            all_opt = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
            grads = all_opt.compute_gradients(objective)
            hist_grad = tf.summary.histogram('gradient_emb', grads[0][0].values)
            optimizer = all_opt.apply_gradients(grads)

    train_skip_sums = tf.summary.merge([hist_out, hist_grad])
    test_sum_ops = [test_loss, accuracy, top_k]

    ops = dict()
    ops['optimizer'] = optimizer
    ops['train_sums'] = tf.summary.merge(tf.get_collection("train_sums"))
    ops['train_skip_sums'] = train_skip_sums
    ops['test_sums'] = test_sum_ops
    ops['test_loss'] = test_loss
    if FLAGS.morph_out:
        ops['out_morph'] = out_morph
    if FLAGS.dropout < 1.:
        ops['dropout_keep'] = dropout_keep
    if FLAGS.dropout_emb:
        ops['dropout_keep_emb'] = dropout_keep_emb

    return input_op, input_morph, target_op, ops, morph_one_size

def run_training():
    print("Creating graph ...")
    time_s = time.time()
    input_op, input_morph, target_op, ops, morph_one_size = create_graph()
    print("... done.", time.time() - time_s)

    print("Initializing session ...")
    time_s = time.time()
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    print("... done.", time.time() - time_s)

    prefix = FLAGS.prefix + '/'
    experiment = FLAGS.experiment
    mkpath(prefix + 'graph_defs/' + experiment)
    saver = tf.train.Saver(max_to_keep=FLAGS.keep_models)
    g_f = open(prefix + 'graph_defs/' + experiment + '/saver_def', 'wb')
    g_f.write(saver.as_saver_def().SerializeToString())
    g_f.close()
    g_f = open(prefix + 'graph_defs/' + experiment + '/graph_def', 'wb')
    g_f.write(tf.get_default_graph().as_graph_def().SerializeToString())
    g_f.close()
    summary_writer = tf.summary.FileWriter(prefix + 'logs/' + experiment,
                                           graph=sess.graph)
    mkpath(prefix + 'models/' + experiment)

    def create_feed_dict(data_all, is_test=False):
        if FLAGS.morph or FLAGS.morph_out:
            data_x, data_x2, data_y = data_all
        else:
            data_y = data_all
        feed_dict = dict()
        feed_dict[target_op] = data_y[:, FLAGS.order - 1]
        if FLAGS.morph:
            feed_dict[input_op] = data_x[:, :FLAGS.order - 1]
            feed_dict[input_morph] = data_x2[:, :(FLAGS.order - 1) * morph_one_size]
        else:
            feed_dict[input_op] = data_y[:, :FLAGS.order - 1]
        if FLAGS.morph_out:
            feed_dict[ops['out_morph']] = data_x2[:, ((FLAGS.order - 1) * morph_one_size):]
        if FLAGS.dropout < 1.:
            feed_dict[ops['dropout_keep']] = 1. if is_test else FLAGS.dropout
        if FLAGS.dropout_emb:
            feed_dict[ops['dropout_keep_emb']] = 1. if is_test else 0.8
        return feed_dict

    perplexities = []
    step = 0
    for it in range(FLAGS.max_iter):
        print("Preparing data ...")
        time_s = time.time()
        if FLAGS.data_train is None or FLAGS.data_dev is None:
            raise Exception("Missing dataset.")
            data = raw_batch_reader(my_open(FLAGS.data_train, 'r'),
                            FLAGS.batch_size,
                            FLAGS.morph or FLAGS.morph_out)
            test_data = raw_batch_reader(my_open(FLAGS.data_dev, 'r'),
                                 FLAGS.batch_size, FLAGS.morph or FLAGS.morph_out)

        print("... done.", time.time() - time_s)

        t_start = time.time()
        for b, (data_all, _) in enumerate(data):
            step += 1
            if (b + 1) % 200 == 0:
                _, sum1, sum2 = sess.run([ops['optimizer'], ops['train_sums'],
                                          ops['train_skip_sums']],
                                         feed_dict=create_feed_dict(data_all))
                summary_writer.add_summary(sum1, step)
                summary_writer.add_summary(sum2, step)
                print("Iteration:", it + 1, "Batch:", b + 1, "Time:", time.time() - t_start)
                t_start = time.time()
            else:
                _, summary_str = sess.run([ops['optimizer'], ops['train_sums']],
                                          feed_dict=create_feed_dict(data_all))
                summary_writer.add_summary(summary_str, step)
        test_loss = 0
        accuracy = 0
        top_k_ac = 0
        total_size = 0
        for (test_data_all, b_s) in test_data:
            t_loss, acc, top_k = sess.run(ops['test_sums'], \
                feed_dict=create_feed_dict(test_data_all, True))
            test_loss += t_loss
            accuracy += acc
            top_k_ac += top_k
            total_size += b_s
        test_loss = test_loss / total_size
        accuracy = accuracy / total_size
        top_k_ac = top_k_ac / total_size
        s_val = lambda t, v: tf.Summary.Value(tag=t, simple_value=v)
        summary_writer.add_summary(
            tf.Summary(value=[
                s_val('loss_test', test_loss),
                s_val('accuracy', accuracy),
                s_val('accuracy_top-10', top_k_ac)
                ]), step)
        perplexities.append(math.exp(test_loss))
        print("\nIteration:", it + 1, "Test loss:", test_loss, \
                "Accuracy:", accuracy, "Perplexity:", perplexities[-1], "\n")
        saver.save(sess, prefix + 'models/' + experiment + '/model', step)

    engine = 'morphlm' if FLAGS.morph else 'formlm'

    print(','.join(map(str, [engine
                    , FLAGS.data_train
                    , FLAGS.max_iter
                    , FLAGS.noise
                    , FLAGS.emb_size
                    , min(perplexities)
                   ])))

def main(_):
    print_info()
    run_training()

if __name__ == '__main__':
    define_args()
    tf.app.run()
