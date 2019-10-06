#include <vector>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>

#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <omp.h>

#include "Python.h"

using namespace std;


inline void Assert(bool is_ok, string error_msg="Assert!") {
  if (!is_ok) {
    cerr << error_msg << endl;
    exit(1);
  }
}


inline void path_join(const char* dir, const char* file, char* obj) {
  if (dir[strlen(dir)-1] == '/')
    sprintf(obj, "%s%s", dir, file);
  else
    sprintf(obj, "%s/%s", dir, file);
}



class GraphReader {

private:
  int64_t N, K;
  vector<vector<vector<int>>> adj;

public:
  GraphReader(int _N, int _K) : N(_N), K(_K) {

    adj.resize(N);
#pragma omp parallel for num_threads(10)
    for (int i = 0; i < N; i++) {
      adj[i].resize(K);
    }
  }

  int read_file(const char* graph_path, int k) {

    int fd = open(graph_path, O_RDONLY);
    Assert(fd != -1, "Open file error.");
    off_t len = lseek(fd, 0, SEEK_END);
    Assert(len >= 0, "lseek file error.");
    char* mbuf = (char*) mmap(NULL, len, PROT_READ, MAP_PRIVATE, fd, 0);

    int line = 0;
    for (const char *p = mbuf, *end = mbuf + len; p < end; line++) {
      char* q;
      int source = strtol(p, &q, 10);
      p = q + 1;
      int target = strtol(q, &q, 10);
      p = q + 1;
      while (p < end && (*p) != '\n') p++;

      adj[source][k].push_back(target);
      adj[target][k].push_back(source);
      p++;
    }

    close(fd);
    munmap((void*)mbuf, len);
    return line;
  }

  void set_len(int* len) {
    int64_t total = N * K;

#pragma omp parallel for num_threads(10)
    for (int64_t iter = 0; iter < total; iter++) {
      int i = iter / K, k = iter % K;
      vector<int>& cell = adj[i][k];

      sort(cell.begin(), cell.end());
      vector<int>::iterator new_end = unique(cell.begin(), cell.end());
      cell.erase(new_end, cell.end());

      len[iter] = cell.size();
    }
  }

  void set_value(const int* idx, const int* len, int* values) {
    int64_t total = N * K;

#pragma omp parallel for num_threads(10)
    for (int64_t iter = 0; iter < total; iter++) {
      int i = iter / K, k = iter % K;
      int n = len[iter], start = idx[iter];
      Assert((size_t) n == adj[i][k].size());

      for (int j = 0; j < n; j++)
        values[start + j] = adj[i][k][j];
    }
  }

};


extern "C" {

GraphReader* new_graph_reader(int N, int K) {
  GraphReader* G = new GraphReader(N, K);
  return G;
}

int read_file(GraphReader* G, const char* graph_path, int k) {
  return G->read_file(graph_path, k);
}

void set_len(GraphReader* G, int* len) {
  G->set_len(len);
}

void set_value(GraphReader* G, int* idx, int* len, int* values) {
  G->set_value(idx, len, values);
}

}
