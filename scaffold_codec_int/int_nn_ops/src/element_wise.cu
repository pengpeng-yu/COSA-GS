#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "gelu_lut.h"

namespace int_nn_ops {

namespace {

constexpr int SharedFxpShift = 20;
constexpr int ONE = 1 << SharedFxpShift;

__inline__ __device__ int64_t rounded_shift(
  const int64_t value,
  const int right_shift)
{
  if (right_shift == 0) return value;
  const int64_t half = int64_t(1) << (right_shift - 1);
  if (value >= 0) return (value + half) >> right_shift;
  else return -((-value + half) >> right_shift);
}

__inline__ __device__ int32_t scalar_gelu(const int32_t input)
{
  if (input >= 6 * ONE) return input;
  if (input <= -6 * ONE) return 0;
  const int32_t magnitude = input < 0 ? -input : input;
  constexpr int fraction_bits = SharedFxpShift - 9;
  const int idx = magnitude >> fraction_bits;
  const int fraction = magnitude & ((1 << fraction_bits) - 1);
  const int64_t value = GELU_LUT[idx];
  const int64_t numerator = (value << fraction_bits) + (int64_t(GELU_LUT[idx + 1]) - value) * fraction;
  return (input > 0 ? input : 0) - static_cast<int32_t>(
    rounded_shift(numerator, 24 + fraction_bits - SharedFxpShift));
}

template <typename T, bool WithGelu, bool WithOutputRequant>
__inline__ __device__ T scalar_requant(
  const int32_t input,
  const uint32_t requant_mul,
  const int right_shift,
  const uint32_t requant_mul_out,
  const int right_shift_out,
  const int64_t zero_point)
{
  int64_t value = rounded_shift(int64_t(input) * requant_mul, right_shift);
  if constexpr (WithGelu) value = scalar_gelu(value);
  if constexpr (WithOutputRequant) value = rounded_shift(value * requant_mul_out, right_shift_out);
  value += zero_point;
  if constexpr (std::is_same_v<T, int8_t>) {
    return static_cast<int8_t>(max(int64_t(-127), min(int64_t(127), value)));
  }
  else {
    return value;
  }
}

template <typename T, bool WithGelu, bool WithOutputRequant>
__global__ void requant_kernel(
  const int32_t* __restrict__ input,
  const uint32_t* __restrict__ requant_mul,
  const int32_t* __restrict__ right_shift,
  const uint32_t* __restrict__ requant_mul_out,
  const int32_t* __restrict__ right_shift_out,
  const int32_t* __restrict__ zero_point,
  T* __restrict__ out,
  const int64_t total,
  const int64_t Ch,
  const int64_t requant_mul_count)
{
  int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = int64_t(blockDim.x) * gridDim.x;

  const int right_shift_val = right_shift[0];
  const uint32_t requant_mul_out_val = WithOutputRequant ? requant_mul_out[0] : 0;
  const int right_shift_out_val = WithOutputRequant ? right_shift_out[0] : 0;
  const int64_t zero_point_val = WithGelu && !WithOutputRequant ? 0 : zero_point[0];

  for (; idx < total; idx += stride) {
    const int64_t ch = idx % Ch;
    const int64_t requant_ch = requant_mul_count == 1 ? 0 : ch;
    out[idx] = scalar_requant<T, WithGelu, WithOutputRequant>(
      input[idx],
      requant_mul[requant_ch],
      right_shift_val,
      requant_mul_out_val,
      right_shift_out_val,
      zero_point_val);
  }
}

__global__ void gelu_kernel(
  const int32_t* __restrict__ input,
  int32_t* __restrict__ out,
  const int64_t total)
{
  int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = int64_t(blockDim.x) * gridDim.x;

  for (; idx < total; idx += stride) {
    out[idx] = scalar_gelu(input[idx]);
  }
}

template <typename T, bool WithGelu, bool WithOutputRequant>
at::Tensor requant(
  const at::Tensor &input,
  const at::Tensor &requant_mul,
  const at::Tensor &right_shift,
  const at::Tensor &requant_mul_out,
  const at::Tensor &right_shift_out,
  const at::Tensor &zero_point,
  at::ScalarType out_dtype)
{
  at::Device device = input.device();
  TORCH_CHECK(device.is_cuda() && input.scalar_type() == at::kInt && input.is_contiguous());
  TORCH_CHECK(requant_mul.device() == device &&
    requant_mul.scalar_type() == at::kUInt32 && requant_mul.is_contiguous());
  TORCH_CHECK(right_shift.device() == device &&
    right_shift.scalar_type() == at::kInt && right_shift.numel() == 1);
  TORCH_CHECK(requant_mul.numel() == 1 || requant_mul.numel() == input.size(-1));
  if constexpr (WithOutputRequant) {
    TORCH_CHECK(requant_mul_out.device() == device &&
      requant_mul_out.scalar_type() == at::kUInt32 && requant_mul_out.numel() == 1);
    TORCH_CHECK(right_shift_out.device() == device &&
      right_shift_out.scalar_type() == at::kInt && right_shift_out.numel() == 1);
  }
  if constexpr (!WithGelu || WithOutputRequant) {
    TORCH_CHECK(zero_point.device() == device &&
      zero_point.scalar_type() == at::kInt && zero_point.numel() == 1);
  }

  c10::cuda::CUDAGuard guard(device);
  at::Tensor out = at::empty(input.sizes(), input.options().dtype(out_dtype));
  const int64_t total = input.numel();
  if (total == 0) return out;

  const int64_t Ch = input.size(-1);
  const int64_t threads = 256;
  const int64_t blocks = std::min<int64_t>((total + threads - 1) / threads, 65535);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());
  requant_kernel<T, WithGelu, WithOutputRequant><<<blocks, threads, 0, stream>>>(
    input.data_ptr<int32_t>(),
    requant_mul.data_ptr<uint32_t>(),
    right_shift.data_ptr<int32_t>(),
    WithOutputRequant ? requant_mul_out.data_ptr<uint32_t>() : nullptr,
    WithOutputRequant ? right_shift_out.data_ptr<int32_t>() : nullptr,
    WithGelu && !WithOutputRequant ? nullptr : zero_point.data_ptr<int32_t>(),
    out.data_ptr<T>(),
    total, Ch, requant_mul.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}

at::Tensor requant_to_int8(
  const at::Tensor &input,
  const at::Tensor &requant_mul,
  const at::Tensor &right_shift,
  const at::Tensor &zero_point)
{
  return requant<int8_t, false, false>(input, requant_mul, right_shift, at::Tensor(), at::Tensor(), zero_point, at::kChar);
}

at::Tensor requant_to_int32(
  const at::Tensor &input,
  const at::Tensor &requant_mul,
  const at::Tensor &right_shift,
  const at::Tensor &zero_point)
{
  return requant<int32_t, false, false>(input, requant_mul, right_shift, at::Tensor(), at::Tensor(), zero_point, at::kInt);
}

at::Tensor gelu(const at::Tensor &input)
{
  at::Device device = input.device();
  TORCH_CHECK(device.is_cuda() && input.scalar_type() == at::kInt && input.is_contiguous());
  c10::cuda::CUDAGuard guard(device);
  at::Tensor out = at::empty_like(input);
  const int64_t total = input.numel();
  if (total == 0) return out;

  const int64_t threads = 256;
  const int64_t blocks = std::min<int64_t>((total + threads - 1) / threads, 65535);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());
  gelu_kernel<<<blocks, threads, 0, stream>>>(
    input.data_ptr<int32_t>(),
    out.data_ptr<int32_t>(),
    total);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor requant_gelu_to_int32(
  const at::Tensor &input,
  const at::Tensor &requant_mul,
  const at::Tensor &right_shift)
{
  return requant<int32_t, true, false>(input, requant_mul, right_shift, at::Tensor(), at::Tensor(), at::Tensor(), at::kInt);
}

at::Tensor requant_gelu_requant_to_int8(
  const at::Tensor &input,
  const at::Tensor &requant_mul,
  const at::Tensor &right_shift,
  const at::Tensor &requant_mul_out,
  const at::Tensor &right_shift_out,
  const at::Tensor &zero_point)
{
  return requant<int8_t, true, true>(input, requant_mul, right_shift,
                                   requant_mul_out, right_shift_out, zero_point, at::kChar);
}

} // namespace int_nn_ops
