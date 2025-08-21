#include <cuda_runtime.h>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/transform.hpp>
#include <fstream>
#include <nvbench/nvbench.cuh>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <vector>

// Benchmark function for cudf::transform uppercase conversion
void bench_transform_uppercase(nvbench::state &state) {
  auto stream = rmm::cuda_stream_view{};

  rmm::mr::cuda_memory_resource cuda_mr{};
  rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> mr{
      &cuda_mr, rmm::percent_of_free_device_memory(50)};
  rmm::mr::set_current_device_resource_ref(mr);

  // Read CSV file on host (do this once, outside the timing loop)
  std::ifstream csv_file("lorem_ipsum.csv");
  if (!csv_file.is_open()) {
    state.skip("Could not open lorem_ipsum.csv");
    return;
  }

  // Vector to store U8 values
  std::vector<uint8_t> char_values;

  // Skip header line
  std::string header;
  std::getline(csv_file, header);

  // Read characters from CSV
  std::string line;
  while (std::getline(csv_file, line)) {
    if (!line.empty()) {
      char_values.push_back(static_cast<uint8_t>(line[0]));
    }
  }
  csv_file.close();

  const size_t num_chars = char_values.size();
  std::cout << "num_chars: " << num_chars << std::endl;

  // If we don't have enough data, repeat what we have
  size_t original_size = char_values.size();
  if (original_size == 0) {
    state.skip("No data read from CSV");
    return;
  }

  // Truncate to exact size
  char_values.resize(original_size);

  // Create cuDF column
  auto column = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::UINT8}, char_values.size(),
      cudf::mask_state::UNALLOCATED, stream, mr);

  // Copy data to device
  cudaMemcpy(column->mutable_view().data<uint8_t>(), char_values.data(),
             char_values.size() * sizeof(uint8_t), cudaMemcpyHostToDevice);

  // Transform expression for uppercase conversion
  std::string transform_expr =
      "__device__ inline void f(uint8_t* output, uint8_t input) { "
      "  *output = ((input >= 97) && (input <= 122)) ? (input - 32) : input; "
      "}";

  std::vector<cudf::column_view> input_columns = {column->view()};

  // Add memory reads/writes to state
  state.add_element_count(num_chars);
  state.add_global_memory_reads<uint8_t>(num_chars);
  state.add_global_memory_writes<uint8_t>(num_chars);

  // Run the benchmark
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch &launch) {
    cudf::transform(input_columns, transform_expr,
                    cudf::data_type{cudf::type_id::UINT8}, false, std::nullopt,
                    rmm::cuda_stream_view{launch.get_stream()}, mr);
  });
}

// Register the benchmark
NVBENCH_BENCH(bench_transform_uppercase);

// Main function for nvbench
NVBENCH_MAIN