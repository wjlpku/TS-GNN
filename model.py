import os
import collections
import numpy as np
import tensorflow as tf
from utils import flatten, repeat

class GraphTuple(
    collections.namedtuple("GraphTuple",
                           ["values", "indices", "lens"])):

  def replace(self, **kwargs):
    return self._replace(**kwargs)


draw_module = tf.load_op_library('./draw.so')

class Model:

  def draw(self, input_nodes, full_graph, max_sample_size):
    n = tf.size(input_nodes)
    cur_idx = tf.gather(full_graph.idx, input_nodes)
    cur_len = tf.gather(full_graph.len, input_nodes)

    cand_sample_size = tf.minimum(max_sample_size, cur_len)
    cand_sample_idx = tf.reshape(
        tf.cumsum(tf.reshape(cand_sample_size, [-1]), exclusive=True),
        [-1, self.K])

    cand_sample_value = draw_module.draw(
        cur_idx, cur_len, full_graph.values, full_graph.J, full_graph.q,
        cand_sample_idx, cand_sample_size)
    return GraphTuple(cand_sample_value, cand_sample_idx, cand_sample_size)


  def __init__(self, sampler, FLAGS, reuse=False):
    self.N, self.K = sampler.N, sampler.K
    self.FLAGS = FLAGS
    self._sampler = sampler
    max_neisize = FLAGS.max_neisize
    H = FLAGS.embedding_dim
    L = FLAGS.layers

    with tf.variable_scope('model', reuse=reuse):

      with tf.name_scope("placeholder"):
        self.inputs = tf.placeholder(tf.int32, (None, 2))
        self.labels = tf.placeholder(tf.float32, (None,))

      with tf.name_scope("embedding"):
        emb_node = tf.get_variable("emb_node", (self.N, H))
        emb_type = tf.get_variable("emb_type", (self.K, H))
        self.bias_table = tf.get_variable(
            'bias_table', (self.N,), initializer=tf.zeros_initializer)
        self.global_bias = tf.get_variable(
            'global_bias', (), initializer=tf.zeros_initializer)
        node_type_weight = [
            tf.get_variable("node_type_weight_" + str(i), shape=(H, H))
            for i in range(L)]
        node_type_att = [None for i in range(L)]

      with tf.name_scope("networks"):
        nodes = [None for i in range(L)]
        cands = [None for i in range(L)]
        nodes.append(tf.reshape(self.inputs, [-1]))

        for i in range(L-1, -1, -1):
          node_type_att[i] = tf.exp(tf.matmul(
              tf.gather(emb_node, nodes[i+1]),
              tf.matmul(node_type_weight[i], emb_type, transpose_b=True)))
          node_type_att[i] *= tf.gather(sampler.node_type_mask, nodes[i+1])
          node_type_att[i] /= tf.reduce_sum(node_type_att[i], axis=1, keepdims=True)
          cands[i] = self.draw(
              nodes[i+1], sampler,
              tf.cast(tf.ceil(node_type_att[i] * max_neisize), tf.int32))
          nodes[i] = tf.concat((nodes[i+1], cands[i].values), axis=-1)


        regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.l2_reg)
        hiddens = [tf.gather(emb_node, nodes[0])]

        for i in range(L):
          with tf.variable_scope("l_%d" % i):
            cur_hidden = tf.layers.dense(
                hiddens[i][ : tf.size(nodes[i+1])],
                H,
                kernel_regularizer=regularizer,
                name='cur')
            nei_hidden = tf.layers.dense(
                hiddens[i][tf.size(nodes[i+1]) : ],
                H,
                kernel_regularizer=regularizer,
                name='nei')
            segment_ids = repeat(
                tf.range(tf.size(cands[i].lens)),
                flatten(cands[i].lens))
            nei_hidden_agg = tf.segment_mean(nei_hidden, segment_ids)
            nei_hidden_agg = tf.reshape(
                tf.concat(
                  [nei_hidden_agg, tf.zeros((tf.size(cands[i].lens)-tf.shape(nei_hidden_agg)[0], H))],
                  axis=0),
                [-1, self.K, H])
            nei_hidden_agg_att = tf.reduce_sum(
                nei_hidden_agg * tf.expand_dims(node_type_att[i], 2),
                axis=1)
            next_hidden = tf.layers.dense(
                tf.concat([cur_hidden, nei_hidden_agg_att], axis=1),
                H,
                kernel_regularizer=regularizer,
                name='nxt')
            hiddens.append(next_hidden)
        output = tf.reshape(hiddens[L], [-1, 2, H])
        bias = tf.gather(self.bias_table, self.inputs)
        self.output = tf.reduce_sum(
            tf.multiply(output[:, 0, :], output[:, 1, :]),
            axis=-1) + bias[:, 0] + bias[:, 1] + self.global_bias

      if reuse: return

      self.saver = tf.train.Saver()

      with tf.name_scope('loss'):
        l2_loss = tf.losses.get_regularization_loss()
        self.cost = tf.losses.mean_squared_error(self.labels, self.output) + l2_loss

      with tf.name_scope("optimizer"):
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        if FLAGS.optimizer == 'adam':
          optimizer = tf.train.AdamOptimizer(FLAGS.lrate)
        elif FLAGS.optimizer == 'ftrl':
          optimizer = tf.train.FtrlOptimizer(FLAGS.lrate)
        else:
          optimizer = tf.train.GradientDescentOptimizer(FLAGS.lrate)
        grads_and_vars = optimizer.compute_gradients(self.cost)
        self.train_op = optimizer.apply_gradients(
            grads_and_vars, global_step=self.global_step)

  def save(self, sess):
    """Save the model to the checkpoint file."""
    checkpoint_path = os.path.join(self.FLAGS.model_name, 'save_model')
    self.saver.save(sess, save_path=checkpoint_path,
                    global_step=self.global_step.eval(),
                    write_meta_graph=False)


  def restore(self, sess, path=None):
    """Restore the model from the checkpoint file."""
    if path is None:
      path = tf.train.latest_checkpoint(self.FLAGS.model_name)
    var_list = tf.trainable_variables()
    saver = tf.train.Saver(var_list)
    saver.restore(sess, save_path=path)
    print('Model restored from %s.' % path, flush=True)
