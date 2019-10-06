#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;


class Random {
public:
  Random(uint32_t seed=123456789) : x(seed) {}

  uint32_t fast_uint32() {
    // https://codingforspeed.com/using-faster-psudo-random-generator-xorshift/
    // https://experilous.com/1/blog/post/perfect-fast-random-floating-point-numbers
    uint32_t t;
    t = x ^ (x << 11);
    x = y; y = z; z = w;
    return w = w ^ (w >> 19) ^ (t ^ (t >> 8));
  }

  uint32_t fast_uniform(uint32_t a=0, uint32_t b=1) {
    uint32_t r = fast_uint32();
    return a + r % (b - a);
  }

  float fast_float() {
    uint32_t r = fast_uint32();
    union {
      uint32_t i;
      float f;
    } pun = { 0x3F800000U | (r >> 9) };
    return pun.f - 1.0f;
  }

  private:
  uint32_t x;
  uint32_t y = 362436069;
  uint32_t z = 521288629;
  uint32_t w = 88675123;
};


REGISTER_OP("Draw")
.Input("cur_idx: int32") // [n, K]
.Input("cur_len: int32") // [n, K]
.Input("full_graph_val: int32") // [N]
.Input("full_graph_j: int32") // [N]
.Input("full_graph_q: float32") // [N]
.Input("cand_sample_idx: int32") // [n, K]
.Input("cand_sample_size: int32") // [n, K]
.Output("output: int32")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
  ::tensorflow::shape_inference::ShapeHandle input;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &input));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &input));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &input));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 1, &input));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 2, &input));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 2, &input));
  c->set_output(0, c->UnknownShapeOfRank(1));
  return Status::OK();
});

class DrawOp : public OpKernel {

private:
  Random generator;

public:
  explicit DrawOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const int32* cur_idx = context->input(0).flat<int32>().data();
    const int32* cur_len = context->input(1).flat<int32>().data();
    const int32* full_graph_val = context->input(2).flat<int32>().data();
    const int32* full_graph_j = context->input(3).flat<int32>().data();
    const float* full_graph_q = context->input(4).flat<float>().data();
    const int32* cand_sample_idx = context->input(5).flat<int32>().data();
    const int32* cand_sample_size = context->input(6).flat<int32>().data();
    const int total = context->input(0).flat<int32>().size();

    TensorShape out_shape;
    out_shape.AddDim(cand_sample_idx[total-1] + cand_sample_size[total-1]);
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape,
          &output_tensor));
    int32* output = output_tensor->flat<int32>().data();

    for (int i = 0; i < total; i++) {
      if (cur_len[i] <= cand_sample_size[i]) {
        memcpy(
            output + cand_sample_idx[i],
            full_graph_val + cur_idx[i],
            cur_len[i] * sizeof(int32));
      } else {
        int n = cand_sample_size[i];

        const int32* cur_value = full_graph_val + cur_idx[i];
        const int32* cur_J = full_graph_j + cur_idx[i];
        const float* cur_q = full_graph_q + cur_idx[i];
        int32* cur_output = output + cand_sample_idx[i];
        const uint32_t sample_range = cur_len[i];

        for (int j = 0; j < n; j++) {
          int kk = generator.fast_uniform(0, sample_range);
          float kk_r = generator.fast_float();
          if (kk_r < cur_q[kk])
            cur_output[j] = cur_value[kk];
          else
            cur_output[j] = cur_value[cur_J[kk]];
        }
      }
    }

    OP_REQUIRES_OK(context, Status::OK());
  }
};

REGISTER_KERNEL_BUILDER(Name("Draw").Device(DEVICE_CPU), DrawOp);
