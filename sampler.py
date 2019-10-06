import os
import ctypes
import numpy as np

alias_lib = ctypes.cdll.LoadLibrary('./alias.so')

alias_lib.build_alias.argtypes = [
  ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
  ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]


reader_lib = ctypes.cdll.LoadLibrary('./reader.so')

reader_lib.new_graph_reader.argtypes = [
  ctypes.c_int, ctypes.c_int]
reader_lib.new_graph_reader.restype = ctypes.c_void_p

reader_lib.read_file.argtypes = [
  ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
reader_lib.read_file.restype = ctypes.c_int

reader_lib.set_len.argtypes = [
  ctypes.c_void_p, ctypes.c_void_p]

reader_lib.set_value.argtypes = [
  ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

class Sampler:

  def __init__(self, dataset_dir, class_type):
    assert class_type in ['train', 'valid', 'test']
    self.class_type = class_type
    print(class_type)

    # Meta Node
    node_type_names, node_type_num = [], []
    with open(os.path.join(dataset_dir, 'meta.txt')) as fin:
      for line in fin:
        line = line.strip().split()
        node_type_names.append(line[0])
        node_type_num.append(int(line[1]))
    print('Graph node meta:', list(zip(node_type_names, node_type_num)),
          flush=True)
    self.node_type_names, self.node_type_num = node_type_names, node_type_num

    # Meta Edge
    edge_filenames = [v
        for v in sorted(os.listdir(dataset_dir))
        if '_' in v and v[0] != '_']
    if class_type == 'train':
      edge_filenames = [v for v in edge_filenames if 'test' not in v]
      edge_filenames = [v for v in edge_filenames if 'valid' not in v]
    elif class_type == 'valid':
      edge_filenames = [v for v in edge_filenames if 'test' not in v]
    print('Graph file:', edge_filenames, flush=True)

    edge_type_names = list(sorted(set([
      '_'.join(v.split('.')[0].split('_')[:3]) for v in edge_filenames])))
    N, K = sum(node_type_num), len(edge_type_names)
    self.N, self.K = N, K
    edge_type_num = [0 for i in range(K)]
    adj = reader_lib.new_graph_reader(N, K)

    for edge_filename in edge_filenames:
      t1_name, t2_name, edge_name = edge_filename.split('.')[0].split('_')[:3]
      t1_idx = node_type_names.index(t1_name)
      t2_idx = node_type_names.index(t2_name)
      k = edge_type_names.index(t1_name + '_' + t2_name + '_' + edge_name)

      edge_filepath = os.path.join(dataset_dir, edge_filename)
      edge_type_num[k] += reader_lib.read_file(
          adj, edge_filepath.encode('utf-8'), k)
    print('Graph edge files:', list(zip(edge_type_names, edge_type_num)),
          flush=True)
    self.edge_type_names, self.edge_type_num = edge_type_names, edge_type_num

    # Build graph structure: idx, len, values
    self.len = np.ones((N, K), np.int32) * -1
    reader_lib.set_len(
        adj, self.len.ctypes.data_as(ctypes.c_void_p))
    assert np.all(self.len != -1)

    self.idx = np.empty((N, K), np.int32)
    self.idx[0][0] = 0
    np.ravel(self.idx)[1:] = np.cumsum(self.len)[:-1]
    assert np.all(np.cumsum(self.len)[:-1] == np.ravel(self.idx)[1:])

    self.values = np.empty((self.idx[-1][-1] + self.len[-1][-1]), np.int32)
    reader_lib.set_value(
        adj,
        self.idx.ctypes.data_as(ctypes.c_void_p),
        self.len.ctypes.data_as(ctypes.c_void_p),
        self.values.ctypes.data_as(ctypes.c_void_p))
    print('Nodes: %d   Edge: %d   K: %d'
        % (N, K, len(self.values)), flush=True)

    # Build alias
    self.J = np.ones_like(self.values, np.int32) * -1
    self.q = np.ones_like(self.values, np.float32) * -1
    alias_lib.build_alias(
        self.idx.ctypes.data_as(ctypes.c_void_p),
        self.len.ctypes.data_as(ctypes.c_void_p),
        self.values.ctypes.data_as(ctypes.c_void_p),
        N, K,
        self.J.ctypes.data_as(ctypes.c_void_p),
        self.q.ctypes.data_as(ctypes.c_void_p))
    assert np.all(self.q > 0)

    # Build rats
    rats_path = None
    for v in os.listdir(dataset_dir):
      if v.endswith('_rats.txt') and class_type in v:
        rats_path = v
        break
    assert rats_path is not None
    ids = []
    labels =[]
    with open(os.path.join(dataset_dir, rats_path)) as f:
      for line in f:
        u, b, rat = line.strip().split()
        ids.append((int(u), int(b)))
        labels.append(float(rat))
    self.ids = np.array(ids, np.int32)
    self.labels = np.array(labels, np.float32)
    print('Labels:', rats_path, len(self.ids), flush=True)


  def sample_batchs(self, batch_size):
    idx = np.arange(0, len(self.ids))
    np.random.shuffle(idx)
    ids = self.ids[idx]
    labels = self.labels[idx]

    for start in range(0, len(ids), batch_size):
      end = min(len(ids), start + batch_size)
      yield start / len(ids), ids[start : end], labels[start : end]


  def print_format(self):
    print(self.class_type)
    print('N: %d   K:%d' % (self.N, self.K))
    print('idx/len:', self.idx.shape)
    print('values/J/q:', self.values.shape)
    print('ids/labels:', self.ids.shape)
    print(flush=True)


  def draw(self, input_graph, max_sample_size):
    J, q = self._J[node_id][type_k], self._q[node_id][type_k]
    n = len(J)
    kk = np.random.randint(low=0, high=n, size=size, dtype=np.int32)
    kk_r = np.random.uniform(size=size)
    idx = np.where(kk_r < q[kk], kk, J[kk]-1)
    return self._adj[node_id][type_k][idx]
