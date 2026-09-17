#include <torch/extension.h>

namespace int_nn_ops {

void cutlass_gemm_int8(const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &);
at::Tensor requant_to_int8(const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &);
at::Tensor requant_to_int32(const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &);
at::Tensor gelu(const at::Tensor &);
at::Tensor requant_gelu_to_int32(const at::Tensor &, const at::Tensor &, const at::Tensor &);
at::Tensor requant_gelu_requant_to_int8(const at::Tensor &, const at::Tensor &, const at::Tensor &,
                                     const at::Tensor &, const at::Tensor &, const at::Tensor &);

} // namespace int_nn_ops

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("cutlass_gemm_int8", &int_nn_ops::cutlass_gemm_int8);
    module.def("requant_to_int8", &int_nn_ops::requant_to_int8);
    module.def("requant_to_int32", &int_nn_ops::requant_to_int32);
    module.def("gelu", &int_nn_ops::gelu);
    module.def("requant_gelu_to_int32", &int_nn_ops::requant_gelu_to_int32);
    module.def("requant_gelu_requant_to_int8", &int_nn_ops::requant_gelu_requant_to_int8);
}
