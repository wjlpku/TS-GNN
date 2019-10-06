import os
import random
import subprocess
from tqdm import tqdm

def getFileLines(filePath):
  return int(subprocess.getoutput('wc -l ' + filePath).split()[0])

class Convert:

  def __init__(self):
    self.meta_type_name = []
    self.meta_type_value = []
    self.history_args = []

  def convert(self, graph_file,
              t1_name, t2_name, edge_name,
              t1_value_type=int, t2_value_type=int,
              sep=None):
  
    if not os.path.exists(graph_file):
      print(graph_file, 'not exists')
      exit()
    print('Converting', graph_file)

    if t1_name not in self.meta_type_name:
      self.meta_type_name.append(t1_name)
      self.meta_type_value.append(set())
    t1_idx = self.meta_type_name.index(t1_name)

    if t2_name not in self.meta_type_name:
      self.meta_type_name.append(t2_name)
      self.meta_type_value.append(set())
    t2_idx = self.meta_type_name.index(t2_name)

    self.history_args.append(
        (graph_file, t1_name, t2_name, edge_name,
         t1_value_type, t2_value_type, sep))

    with open(graph_file, 'r') as fin:
      for line in tqdm(fin, total=getFileLines(graph_file)):
        line = line.strip().split(sep)
        assert len(line) == 2 or len(line) == 3, line

        t1_value, t2_value = t1_value_type(line[0]), t2_value_type(line[1])
        self.meta_type_value[t1_idx].add(t1_value)
        self.meta_type_value[t2_idx].add(t2_value)


  def _transfer(self, target_dir, graph_file,
                t1_name, t2_name, edge_name,
                t1_value_type=int, t2_value_type=int,
                sep=None):
 
    if not os.path.exists(graph_file):
      print(graph_file, 'not exists')
      exit()

    t1_idx = self.meta_type_name.index(t1_name)
    t1_list = self.meta_type_value[t1_idx]
    t1_idx_start = sum([len(v) for v in self.meta_type_value[:t1_idx]])
    t1_dict = dict(zip(t1_list, range(t1_idx_start, t1_idx_start + len(t1_list))))

    t2_idx = self.meta_type_name.index(t2_name)
    t2_list = self.meta_type_value[t2_idx]
    t2_idx_start = sum([len(v) for v in self.meta_type_value[:t2_idx]])
    t2_dict = dict(zip(t2_list, range(t2_idx_start, t2_idx_start + len(t2_list))))

    file_name = '%s_%s_%s.txt' % (t1_name, t2_name, edge_name)
    print('Transfering', graph_file, 'to', file_name)
    target_path = os.path.join(target_dir, file_name)

    with open(graph_file, 'r') as fin, open(target_path, 'w') as fout:
      for line in tqdm(fin, total=getFileLines(graph_file)):
        line = line.strip().split(sep)

        t1_value, t2_value = t1_value_type(line[0]), t2_value_type(line[1])
        t1_remap_idx, t2_remap_idx = t1_dict[t1_value], t2_dict[t2_value]
        print(t1_remap_idx, t2_remap_idx, end=' ', file=fout)
        rat_value = line[2] if len(line) == 3 else '1'
        print(rat_value, file=fout)

  def transfer(self, target_dir):
    self.meta_type_value = [list(sorted(v)) for v in self.meta_type_value]

    i = 0
    with open(os.path.join(target_dir, 'meta.txt'), 'w') as fout_meta, \
        open(os.path.join(target_dir, 'remap.txt'), 'w') as fout_remap:
      for type_name, type_value in zip(self.meta_type_name, self.meta_type_value):
        print(type_name, len(type_value), file=fout_meta)
        print(type_name, len(type_value))
        for v in type_value:
          print(v, i, file=fout_remap)
          i += 1

    for args in self.history_args:
      self._transfer(target_dir, *args)


  def _print_rats(self, dat, file_path):
    with open(file_path, 'w') as f:
      for d in dat:
        print(d, end='', file=f)
      print(file_path, len(dat), sep='\t')

  def split2(self, target_dir, target_filename,
             train_part=0.7, valid_part=0.1, seed=1234):

    print('Spliting', target_filename)
    random.seed(seed)

    with open(os.path.join(target_dir, target_filename)) as f:
      dat = f.readlines()
      random.shuffle(dat)

    if target_filename[0] != '_':
      os.rename(
          os.path.join(target_dir, target_filename),
          os.path.join(target_dir, '_' + target_filename))

    prefix = target_filename.split('.')[0]

    n = len(dat)
    self._print_rats(dat[ : int(train_part * n)],
                     os.path.join(target_dir, prefix + '_train_rats.txt'))
    self._print_rats(dat[int(train_part * n) : int((train_part + valid_part) * n)],
                     os.path.join(target_dir, prefix + '_valid_rats.txt'))
    self._print_rats(dat[int((train_part + valid_part) * n) : ],
                     os.path.join(target_dir, prefix + '_test_rats.txt'))
