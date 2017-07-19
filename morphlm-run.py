#!/usr/bin/env python3

import math

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.core.protobuf import saver_pb2

import morphlm

flags = tf.app.flags
FLAGS = flags.FLAGS

sess = tf.Session()
gt = sess.graph.get_tensor_by_name

def define_args():
    flags.DEFINE_string('data', None, 'Data.')
    flags.DEFINE_string('vocab-forms', None, 'Vocabulary for forms.')
    flags.DEFINE_string('vocab-lemmata', None, 'Vocabulary for lemmata.')
    flags.DEFINE_integer('order', 5, 'N-gram order.')
    flags.DEFINE_boolean('full-tags', False, 'Use longer binary reprezentation for tags')
    flags.DEFINE_string('prefix', './', 'Path prefix.')
    flags.DEFINE_string('experiment', 'test', 'Experiment name.')
    flags.DEFINE_boolean('morph', True, 'Use morphology.')
    flags.DEFINE_boolean('no-morph-input', False, '')

START_WORD = "<s>|<s>|---------------"
END_WORD = "</s>|</s>|Z#-------------"

def ngram_gen(data_file):
    for line in data_file:
        words = (FLAGS.order - 1) * [START_WORD] + line.split() + [END_WORD]
        i = FLAGS.order
        o = FLAGS.order
        l = len(words)
        line_grams = []
        while i <= l:
            line_grams.append(words[i - o:i])
            i += 1
        if len(line_grams) == 0:
            print(words)
        yield line_grams


def batch_reader(data_file):
    ngrams = ngram_gen(data_file)
    c = 0
    data_x = []
    data_x2 = []
    data_y = []
    
    vocab_f = {x.strip() : i for i, x in enumerate(morphlm.my_open(FLAGS.vocab_forms).readlines())}
    vocab_l = {x.strip() : i for i, x in enumerate(morphlm.my_open(FLAGS.vocab_lemmata).readlines())}

    if FLAGS.full_tags:
        mk_tag = morphlm.make_full_tag
    else:
        mk_tag = morphlm.make_tag

    for ngram in ngrams:
        for n in ngram:
            factors = [x.split('|') for x in n]
            forms = [vocab_f.get(x[0], 0) for x in factors]
            data_y.append(forms)
            lemmata = [vocab_l.get(x[1], 0) for x in factors]
            tags = sum([mk_tag(x[2]) for x in factors], [])
            data_x.append(lemmata)
            data_x2.append(tags)
        if FLAGS.morph:
            yield (np.array(data_x), np.array(data_x2, dtype=np.float32),
                   np.array(data_y)), len(ngram)
        else:
            yield np.array(data_y), len(ngram) 
        data_y = []
        data_x = []
        data_x2 = []

def load_graph(graph_path):
    with gfile.FastGFile(graph_path,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

def load_model(saver_path, model_path):
    with gfile.FastGFile(saver_path,'rb') as f:
        saver_def = saver_pb2.SaverDef()
        saver_def.ParseFromString(f.read())
        saver = tf.train.Saver(saver_def=saver_def)
    saver.restore(sess, model_path)

def init(graph_path, saver_path, model_path):
    #print("morphLM init")
    #print("g:", graph_path)
    #print("s:", saver_path)
    #print("m:", model_path)
    load_graph(graph_path)
    #print("graph loaded")
    load_model(saver_path, model_path)

def init_default():
    g = FLAGS.prefix + '/graph_defs/' + FLAGS.experiment + '/graph_def'
    s = FLAGS.prefix + '/graph_defs/' + FLAGS.experiment + '/saver_def'
    ch = open(FLAGS.prefix +
              '/models/' + FLAGS.experiment + '/checkpoint').readlines()[0].split()[1][1:-1]
    m = FLAGS.prefix + '/models/' + FLAGS.experiment  + ch
    m = FLAGS.prefix + '/models/' + FLAGS.experiment + '/' + ch
    init(g, s, m)

def create_feed_dict(data_all, is_test=False):
    if FLAGS.full_tags:
        morph_one_size = 139
    else:
        morph_one_size = 60
    if FLAGS.morph:
        data_x, data_x2, data_y = data_all
    else:
        data_y = data_all
    feed_dict = dict()
    feed_dict['targets/Placeholder:0'] = data_y[:, FLAGS.order - 1]
    if FLAGS.morph:
        feed_dict['inputs/Placeholder:0'] = data_x[:, :FLAGS.order - 1]
        feed_dict['inputs/Placeholder_1:0'] = data_x2[:, :(FLAGS.order - 1) * morph_one_size]
    else:
        feed_dict['inputs/Placeholder:0'] = data_y[:, :FLAGS.order - 1]
    return feed_dict

def main(_):
    init_default()
    # read the data
    data=batch_reader(morphlm.my_open(FLAGS.data, 'r'))
    # go through data and compute things
    for (data_all, b_s) in data:
        logits = \
        sess.run(gt('cross-entropy_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:0'),
            feed_dict=create_feed_dict(data_all))
        print(-sum(logits), len(logits))

if __name__ == '__main__':
    define_args()
    tf.app.run()
