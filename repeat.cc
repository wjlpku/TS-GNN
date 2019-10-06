#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;


REGISTER_OP("Repeat")
.Input("tensor: int32") // [n]
.Input("repeats: int32") // [n]
.Output("output: int32")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
  ::tensorflow::shape_inference::ShapeHandle input;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &input));
  c->set_output(0, c->UnknownShapeOfRank(1));
  return Status::OK();
});

class RepeatOp : public OpKernel {
public:
  explicit RepeatOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const int32* tensor = context->input(0).flat<int32>().data();
    const int32* repeats = context->input(1).flat<int32>().data();
    const int n = context->input(0).flat<int32>().size();

    size_t out_dim = 0;
    for (int i = 0; i < n; i++) {
      OP_REQUIRES(context, repeats[i] >= 0,
          errors::InvalidArgument("Repeats must be positive or zero."));
      out_dim += repeats[i];
    }
    TensorShape out_shape;
    out_shape.AddDim(out_dim);
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape,
          &output_tensor));
    int32* output = output_tensor->flat<int32>().data();

    size_t p = 0;
    for (int i = 0; i < n; i++)
      for (int j = 0; j < repeats[i]; j++)
        output[p++] = tensor[i];
    
    OP_REQUIRES(context, p == out_dim,
        errors::InvalidArgument("Wrong Repeat."));

    OP_REQUIRES_OK(context, Status::OK());
  }
};

REGISTER_KERNEL_BUILDER(Name("Repeat").Device(DEVICE_CPU), RepeatOp);
