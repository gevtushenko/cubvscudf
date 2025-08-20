#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <iostream>
#include <memory>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <tuple>
#include <vector>

int main() {
  int num_items = 1 << 28;
  auto stream = rmm::cuda_stream_view{};
  auto mr = cudf::get_current_device_resource_ref();

  // Create scalar objects for init and step values
  auto init_scalar = cudf::numeric_scalar<int32_t>(0);
  auto step_scalar = cudf::numeric_scalar<int32_t>(1);

  auto result = cudf::sequence(num_items, init_scalar, step_scalar, stream, mr);
  std::cout << result->size() << std::endl;
  return 0;
}
