#include <cassert>
#include <iostream>
#include <sstream>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "src/ssd_update.h"
#include "src/metal_utils.h"

#ifdef ACCELERATE_NEW_LAPACK
#include <vecLib/cblas_new.h>
#endif

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"

namespace mlx::core {

std::vector<array> ssd_update(
    const array& x,
    const array& dt,
    const array& decay,
    const array& B,
    const array& C,
    const array& D,
    const array& z,
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
      std::make_shared<SSDUpdate>(to_stream(s)),
      {x, dt, decay, B, C, D, z, state});
}

void SSDUpdate::eval(const std::vector<array>& inputs,  std::vector<array>& outputs) {
    throw std::runtime_error("eval not implemented!");
}


#ifdef ACCELERATE_NEW_LAPACK

void SSDUpdate::eval_cpu(const std::vector<array>& inputs,  std::vector<array>& outputs) {
    throw std::runtime_error("eval_cpu not implemented!");
}

#endif


void SSDUpdate::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {

  assert(inputs.size() == 8);
  assert(outputs.size() == 2);

  auto& x = inputs[0];
  auto& dt = inputs[1];
  auto& decay = inputs[2];
  auto& B = inputs[3];
  auto& C = inputs[4];
  auto& D = inputs[5];
  auto& z = inputs[6];
  auto& state = inputs[7];

  auto& y = outputs[0];
  auto& next_state = outputs[1];

  auto& s = stream();
  auto& d = metal::device(s.device);

  y.set_data(allocator::malloc(y.nbytes()));
  next_state.set_data(allocator::malloc(next_state.nbytes()));

  std::ostringstream kname;
  kname << "ssd_update_kernel_";
  kname << type_to_name(x);

  auto lib = d.get_library("mlx_ext", cartesia::mlx_ext::metallib_dir());
  auto kernel = d.get_kernel(kname.str(), lib);
  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_input_array(x, 0);
  compute_encoder.set_input_array(dt, 1);
  compute_encoder.set_input_array(decay, 2);
  compute_encoder.set_input_array(B, 3);
  compute_encoder.set_input_array(C, 4);
  compute_encoder.set_input_array(D, 5);
  compute_encoder.set_input_array(z, 6);
  compute_encoder.set_input_array(state, 7);
  compute_encoder.set_output_array(y, 8);
  compute_encoder.set_output_array(next_state, 9);


  auto b = x.shape(0);
  auto h = x.shape(1);
  auto dh = x.shape(2);
  auto g = B.shape(1);
  int group_size = static_cast<int>(h / g);
  int state_size = static_cast<int>(state.shape(3));

  compute_encoder.set_bytes(group_size, 10);
  compute_encoder.set_bytes(state_size, 11);
  compute_encoder.set_vector_bytes(x.strides(), 12);
  compute_encoder.set_vector_bytes(dt.strides(), 13);
  compute_encoder.set_vector_bytes(B.strides(), 14);
  compute_encoder.set_vector_bytes(state.strides(), 15);

  // https://developer.apple.com/documentation/metal/compute_passes/calculating_threadgroup_and_grid_sizes
  MTL::Size grid_dims = MTL::Size(b, h, dh);
  // size_t width = kernel->threadExecutionWidth();
  // size_t height = kernel->maxTotalThreadsPerThreadgroup() / width; 
  MTL::Size group_dims = MTL::Size(32, 32, 1);
  
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

} // namespace mlx::core
