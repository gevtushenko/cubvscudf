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
#include <iostream>
#include <cctype>
#include <algorithm>

// Benchmark function for cudf::transform uppercase conversion
void bench_transform_uppercase(nvbench::state &state) {
  auto stream = rmm::cuda_stream_view{};

  rmm::mr::cuda_memory_resource cuda_mr{};
  rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> mr{
      &cuda_mr, rmm::percent_of_free_device_memory(50)};
  rmm::mr::set_current_device_resource_ref(mr);

  // Get the number of elements from the axis parameter
  const auto num_elements = static_cast<size_t>(state.get_int64("Elements"));
  
  // Read CSV file on host
  std::ifstream csv_file("lorem_ipsum.csv");
  if (!csv_file.is_open()) {
    state.skip("Could not open lorem_ipsum.csv");
    return;
  }

  // Vector to store original U8 values
  std::vector<uint8_t> original_chars;

  // Skip header line
  std::string header;
  std::getline(csv_file, header);

  // Read characters from CSV
  std::string line;
  while (std::getline(csv_file, line)) {
    if (!line.empty()) {
      original_chars.push_back(static_cast<uint8_t>(line[0]));
    }
  }
  csv_file.close();

  if (original_chars.empty()) {
    state.skip("No data read from CSV");
    return;
  }

  // Repeat the data to reach desired element count
  std::vector<uint8_t> char_values;
  char_values.reserve(num_elements);
  
  for (size_t i = 0; i < num_elements; ++i) {
    char_values.push_back(original_chars[i % original_chars.size()]);
  }

  const size_t num_chars = char_values.size();
  std::cout << "num_chars: " << num_chars << std::endl;

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

  // Variable to store the result column
  std::unique_ptr<cudf::column> result_column;
  
  // Run the benchmark
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch &launch) {
    result_column = cudf::transform(input_columns, transform_expr,
                    cudf::data_type{cudf::type_id::UINT8}, false, std::nullopt,
                    rmm::cuda_stream_view{launch.get_stream()}, mr);
  });
  
  // Verify the output after benchmark
  if (result_column) {
    std::vector<uint8_t> output_values(num_chars);
    cudaMemcpy(output_values.data(), result_column->view().data<uint8_t>(), 
               num_chars * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    
    // Check a sample of values to verify uppercase transformation
    bool verification_passed = true;
    int errors_found = 0;
    const int max_errors_to_report = 10;
    
    for (size_t i = 0; i < num_chars && errors_found < max_errors_to_report; ++i) {
      uint8_t input_val = char_values[i];
      uint8_t output_val = output_values[i];
      uint8_t expected_val = ((input_val >= 97) && (input_val <= 122)) ? (input_val - 32) : input_val;
      
      if (output_val != expected_val) {
        if (errors_found == 0) {
          std::cout << "Verification errors found:" << std::endl;
        }
        std::cout << "  Index " << i << ": input=" << static_cast<char>(input_val) 
                  << " (" << static_cast<int>(input_val) << "), output=" << static_cast<char>(output_val)
                  << " (" << static_cast<int>(output_val) << "), expected=" << static_cast<char>(expected_val)
                  << " (" << static_cast<int>(expected_val) << ")" << std::endl;
        verification_passed = false;
        errors_found++;
      }
    }
    
    if (verification_passed) {
      std::cout << "✓ Verification PASSED: All " << num_chars << " characters correctly transformed to uppercase" << std::endl;
    } else {
      std::cout << "✗ Verification FAILED: Found transformation errors" << std::endl;
      if (errors_found >= max_errors_to_report) {
        std::cout << "  (showing first " << max_errors_to_report << " errors only)" << std::endl;
      }
    }
    
    // Show a sample of the transformed text
    std::cout << "Sample of transformed text (first 100 chars):" << std::endl;
    std::cout << "  Input:  ";
    for (size_t i = 0; i < std::min(size_t(100), num_chars); ++i) {
      char c = static_cast<char>(char_values[i]);
      std::cout << (isprint(c) ? c : '.');
    }
    std::cout << std::endl;
    
    std::cout << "  Output: ";
    for (size_t i = 0; i < std::min(size_t(100), num_chars); ++i) {
      char c = static_cast<char>(output_values[i]);
      std::cout << (isprint(c) ? c : '.');
    }
    std::cout << std::endl;
  }
}

// Register the benchmark with Elements axis
NVBENCH_BENCH(bench_transform_uppercase)
  .add_int64_power_of_two_axis("Elements", {20, 24, 28});

// Main function for nvbench
NVBENCH_MAIN