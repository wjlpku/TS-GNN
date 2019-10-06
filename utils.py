import json
import tensorflow as tf
from tensorflow.python.client import timeline


def flatten(x):
  return tf.reshape(x, [-1])


class TimeLiner:
  _timeline_dict = None

  def update_timeline(self, chrome_trace):
    # convert crome trace to python dict
    chrome_trace_dict = json.loads(chrome_trace)
    # for first run store full trace
    if self._timeline_dict is None:
      self._timeline_dict = chrome_trace_dict
    # for other - update only time consumption, not definitions
    else:
      for event in chrome_trace_dict['traceEvents']:
        # events time consumption started with 'ts' prefix
        if 'ts' in event:
          self._timeline_dict['traceEvents'].append(event)

  def save(self, f_name):
    with open(f_name, 'w') as f:
      json.dump(self._timeline_dict, f)


repeat_module = tf.load_op_library('./repeat.so')

def repeat(tensor, repeats, name="repeat"):
  tensor = tf.convert_to_tensor(tensor)
  repeats = tf.convert_to_tensor(repeats)
  with tf.name_scope(name):
    return repeat_module.repeat(tensor, repeats)

class TestRepeat(tf.test.TestCase):

  def test_case(self):
    with self.session() as sess:
      ret = repeat([3, 5, -2], [0, 4, 1])
      self.assertAllEqual(sess.run(ret), [5, 5, 5, 5, -2])


if __name__ == '__main__':
  import os
  os.environ["CUDA_VISIBLE_DEVICES"]='0'
  tf.test.main()
