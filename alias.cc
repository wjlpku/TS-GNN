#include <cmath>
#include <queue>
#include <vector>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <algorithm>

#include <omp.h>
#include <assert.h>
#include "Python.h"

using namespace std;


inline void Assert(bool is_ok, string error_msg="Assert!") {
  if (!is_ok) {
    cerr << error_msg << endl;
    exit(1);
  }
}


void alias_setup(const float* probs, int32_t n, int32_t* J, float* q) {
  queue<int32_t> smaller;
  queue<int32_t> larger;

  for (int i = 0; i < n; i++){
    q[i] = probs[i] * n;
    if (q[i] < 1.0)
      smaller.push(i);
    else
      larger.push(i);
  }

  int small, large;
  while (smaller.size() > 0 && larger.size() > 0) {
    small = smaller.front();
    large = larger.front();
    smaller.pop();
    larger.pop();

    J[small] = large;
    q[large] = q[large] + q[small] - 1.0f;

    if (q[large] < 1.0)
      smaller.push(large);
    else
      larger.push(large);
  }
}

void _build_alias(
    const int32_t* idx, const int32_t* len, const int32_t* values, 
    int N, int K, int32_t* J, float* q) {

  int total = N * K;
#pragma omp parallel for num_threads(10)
  for (int p = 0; p < total; p++) {
    int k = p % K;
    int beg = idx[p], n = len[p];
    if (n == 0) continue;

    float* probs = new float [n];
    float sum = 0.0f;
    for (int iter = 0; iter < n; iter++) {
      int j = values[iter + beg];
      probs[iter] = 1.0f / sqrtf(len[p] * len[j * K + k]);
      sum += probs[iter];
    }
    for (int iter = 0; iter < n; iter++)
      probs[iter] /= sum;
    alias_setup(probs, n, J + beg, q + beg);
    delete probs;
  }
}


extern "C" {
void build_alias(
    const int32_t* idx, const int32_t* len, const int32_t* values, 
    int N, int K, int32_t* J, float* q) {
  _build_alias(idx, len, values, N, K, J, q);
}
}
