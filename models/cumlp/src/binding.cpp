#include <torch/extension.h>


void mlp_fused32(
    int DEPTH, int act_mode,
    at::Tensor inputs,
    at::Tensor outputs,
    at::Tensor params
);

void mlp_fused16(
    int DEPTH, int act_mode,
    at::Tensor inputs,
    at::Tensor outputs,
    at::Tensor params
);

void mlp_reparam16(
    at::Tensor inputs,
    at::Tensor outputs,
    at::Tensor params
);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mlp_fused32", &mlp_fused32);
    m.def("mlp_fused16", &mlp_fused16);
    m.def("mlp_reparam16", &mlp_reparam16);
}