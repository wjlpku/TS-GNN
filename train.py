import os
import time
import json
import pickle
import numpy as np
import tensorflow as tf

from model import Model
from sampler import Sampler
from utils import timeline, TimeLiner

# Environment parameters
tf.flags.DEFINE_string("model_name", "test", "Model name")
tf.flags.DEFINE_string("gpus", "0", "GPUs")
tf.flags.DEFINE_string("dataset", "./graph", "Dataset name")

# Network parameters
tf.flags.DEFINE_integer("embedding_dim", 32, "Dimensionality of embedding")
tf.flags.DEFINE_integer("layers", 2, "Layers of aggregation")
tf.flags.DEFINE_integer("max_neisize", 15, "Layers of aggregation")

# Optimazing parameters
tf.flags.DEFINE_string('optimizer', 'adam', 'Optimizer for training: (adam, ftrl, sgd*)')
tf.flags.DEFINE_float("lrate", 0.0001, "Learning rate")
tf.flags.DEFINE_float("l2_reg", 0.0001, "L2 regulation rate")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch size")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epoches")
tf.flags.DEFINE_integer("eval_interval", 100, "Number of eval interval")
tf.flags.DEFINE_integer("patience", 20, "The number of intevals to wait before early stop if no progress on the validation set.")

FLAGS = tf.flags.FLAGS


def load_dataset(dataset_dir):

  dataset_name = FLAGS.dataset.strip('/').split('/')[-2]
  cache_path = 'data_%s.pkl' % dataset_name

  if os.path.exists(cache_path):
    with open(cache_path, 'rb') as f:
      train_sampler = pickle.load(f)
      valid_sampler = pickle.load(f)
      test_sampler = pickle.load(f)
  else:
    train_sampler = Sampler(dataset_dir, 'train')
    valid_sampler = Sampler(dataset_dir, 'valid')
    test_sampler = Sampler(dataset_dir, 'test')
    with open(cache_path, 'wb') as f:
      pickle.dump(train_sampler, f, pickle.HIGHEST_PROTOCOL)
      pickle.dump(valid_sampler, f, pickle.HIGHEST_PROTOCOL)
      pickle.dump(test_sampler, f, pickle.HIGHEST_PROTOCOL)
    print('Save to', cache_path)

  def wrap(sampler):
    sampler.node_type_mask = tf.convert_to_tensor(
        np.where(sampler.len != 0, 1.0, 0.0), dtype=tf.float32)
    sampler.idx = tf.convert_to_tensor(sampler.idx, tf.int32)
    sampler.len = tf.convert_to_tensor(sampler.len, tf.int32)
    sampler.values = tf.convert_to_tensor(sampler.values, tf.int32)
    sampler.J = tf.convert_to_tensor(sampler.J, tf.int32)
    sampler.q = tf.convert_to_tensor(sampler.q, tf.float32)
    return sampler

  train_sampler.print_format()
  valid_sampler.print_format()
  test_sampler.print_format()

  return wrap(train_sampler), wrap(valid_sampler), wrap(test_sampler)


def eval_dataset(model, sampler, sess):
  outs, ys = [], []
  total_time = time.time()

  for _, batch_data, batch_label in sampler.sample_batchs(FLAGS.batch_size):
    outs.append(sess.run(model.output, feed_dict={
      model.inputs: batch_data}))
    ys.append(batch_label)

  outs = np.concatenate(outs)
  ys = np.concatenate(ys)
  mse = np.fabs(outs - ys).mean()
  rmse = np.sqrt(np.mean((outs - ys)**2))
  total_time = time.time() - total_time

  return rmse, mse, total_time


if __name__ == '__main__':
  os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpus
  np.random.seed(1234)
  tf.set_random_seed(1234)
  print(json.dumps(
    dict({key: getattr(FLAGS, key) for key in FLAGS.__flags.keys()}),
    indent=4))

  with tf.Session(
      config=tf.ConfigProto(
        gpu_options=tf.GPUOptions(allow_growth=True),
        )) as sess:

    train_sampler, valid_sampler, test_sampler = \
        load_dataset(os.environ['HOME'] + FLAGS.dataset)
    train_model = Model(train_sampler, FLAGS)
    valid_model = Model(valid_sampler, FLAGS, reuse=True)
    test_model = Model(test_sampler, FLAGS, reuse=True)
    sess.run(tf.global_variables_initializer())

    for v in tf.trainable_variables():
      print(v)
    print(train_model.cost, flush=True)

    step, loss = 0, float('inf')
    train_time = time.time()
    best_valid, epoch_patience = float('inf'), 0

    # options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    # run_metadata = tf.RunMetadata()
    # many_runs_timeline = TimeLiner()

    end_flag = False
    for epoch in range(FLAGS.num_epochs):
      if end_flag: break

      for ratio, batch_data, batch_label in train_sampler.sample_batchs(
          FLAGS.batch_size):

        if step % FLAGS.eval_interval == 0:

          train_time = time.time() - train_time
          valid_rmse, valid_mse, valid_time = eval_dataset(
              valid_model, valid_sampler, sess)
          print(
              "Epoch: %d (%05.2f%%)   Step: %d   "
              "Train_Loss: %.4f   Train_Time: %.2fs   "
              "Valid_RMSE: %.4f   Valid_MSE: %.4f   valid_time: %.2fs"
              % (epoch, ratio*100, step,
                 loss, train_time,
                 valid_rmse, valid_mse, valid_time), end='')

          if valid_rmse < best_valid:
            save_time = time.time()
            train_model.save(sess)
            save_time = time.time() - save_time
            print('   Save_time: %.2fs' % save_time, end='')
            best_valid = valid_rmse
            epoch_patience = 0
          print(flush=True)

          if epoch_patience >= FLAGS.patience:
            end_flag = True
            break
          epoch_patience += 1

          train_time = time.time()

        # One training step.
        _, loss, step = sess.run(
            [train_model.train_op, train_model.cost, train_model.global_step],
            feed_dict={train_model.inputs: batch_data, train_model.labels: batch_label})
      #       options=options,
      #       run_metadata=run_metadata)

      #   if step < 5:
      #     fetched_timeline = timeline.Timeline(run_metadata.step_stats)
      #     chrome_trace = fetched_timeline.generate_chrome_trace_format()
      #     many_runs_timeline.update_timeline(chrome_trace)
      #     many_runs_timeline.save('profile.json')
      #     break
      # break


    all_paths = tf.train.get_checkpoint_state(FLAGS.model_name).all_model_checkpoint_paths

    for path in all_paths:
      test_model.restore(sess, path=path)
      test_rmse, test_mse, _ = eval_dataset(test_model, test_sampler, sess)
      print("Test_RMSE: %.4f   Test_MSE: %.4f" %
          (test_rmse, test_mse), flush=True)
