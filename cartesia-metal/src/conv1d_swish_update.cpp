#include <cassert>
#include <iostream>
#include <sstream>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "src/conv1d_swish_update.h"
#include "src/metal_utils.h"

#ifdef ACCELERATE_NEW_LAPACK
#include <vecLib/cblas_new.h>
#endif

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"

namespace mlx::core {

std::vector<array> conv1d_swish_update(
    const array& x,
    const array& w, 
    const array& b, 
    const array& state,
    StreamOrDevice s /* = {} */ // Stream on which to schedule the operation
) {
  auto y_dtype = x.dtype();
  // TODO: Also make sure state is of the same dtype

  auto y_shape = x.shape();
  auto state_shape = state.shape();

  return array::make_arrays(
      {y_shape, state_shape},
      {y_dtype, y_dtype},
      std::make_shared<Conv1dSwishUpdate>(to_stream(s)),
      {x, w, b, state});
}


void Conv1dSwishUpdate::eval(const std::vector<array>& inputs,  std::vector<array>& outputs) {
    throw std::runtime_error("eval not implemented!");
}


#ifdef ACCELERATE_NEW_LAPACK

void Conv1dSwishUpdate::eval_cpu(const std::vector<array>& inputs,  std::vector<array>& outputs) {
    throw std::runtime_error("eval_cpu not implemented!");
}

#endif


void Conv1dSwishUpdate::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {

  assert(inputs.size() == 4);
  assert(outputs.size() == 2);

  auto& x = inputs[0];
  auto& w = inputs[1];
  auto& b = inputs[2];
  auto& state = inputs[3];

  auto& y = outputs[0];
  auto& next_state = outputs[1];

  auto& s = stream();
  auto& d = metal::device(s.device);

  y.set_data(allocator::malloc(y.nbytes()));
  next_state.set_data(allocator::malloc(next_state.nbytes()));

  std::ostringstream kname;
  kname << "conv1d_swish_update_kernel_";
  kname << type_to_name(x);

  auto lib = d.get_library("mlx_ext", cartesia::mlx_ext::metallib_dir());
  auto kernel = d.get_kernel(kname.str(), lib);
  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(kernel);

  int kernel_size = static_cast<int>(w.shape(1));

  compute_encoder.set_input_array(x, 0);
  compute_encoder.set_input_array(w, 1);
  compute_encoder.set_input_array(b, 2);
  compute_encoder.set_input_array(state, 3);
  compute_encoder.set_output_array(y, 4);
  compute_encoder.set_output_array(next_state, 5);
  compute_encoder.set_bytes(kernel_size, 6);
  compute_encoder.set_vector_bytes(x.strides(), 7);
  compute_encoder.set_vector_bytes(state.strides(), 8);
  

  auto batch_size = x.shape(0);
  auto n_channels = x.shape(1);

  // https://developer.apple.com/documentation/metal/compute_passes/calculating_threadgroup_and_grid_sizes
  MTL::Size grid_dims = MTL::Size(batch_size, n_channels, 1);
  size_t width = kernel->threadExecutionWidth();
  size_t height = kernel->maxTotalThreadsPerThreadgroup() / width; 
  MTL::Size group_dims = MTL::Size(width, height, 1);
  
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

} // namespace mlx::core
