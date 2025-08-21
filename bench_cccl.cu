#include <algorithm>
#include <cccl/c/transform.h>
#include <cccl/c/types.h>
#include <cctype>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <nvbench/nvbench.cuh>
#include <string>
#include <vector>

// Helper to create iterator for device pointer
cccl_iterator_t make_device_pointer_iterator(void *ptr,
                                             cccl_type_enum value_type) {
  cccl_iterator_t iter;
  iter.size = sizeof(void *);
  iter.alignment = alignof(void *);
  iter.type = cccl_iterator_kind_t::CCCL_POINTER;
  iter.advance.type = cccl_op_kind_t::CCCL_PLUS;
  iter.dereference.type = cccl_op_kind_t::CCCL_IDENTITY;
  iter.value_type.size = sizeof(uint8_t);
  iter.value_type.alignment = alignof(uint8_t);
  iter.value_type.type = value_type;
  iter.state = ptr;
  iter.host_advance = nullptr;
  return iter;
}

// Benchmark function for CCCL C API transform uppercase conversion
void bench_cccl_transform_uppercase(nvbench::state &state) {
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

  if (num_chars == 0) {
    state.skip("No data read from CSV");
    return;
  }

  // Allocate device memory
  uint8_t *d_input;
  uint8_t *d_output;
  cudaMalloc(&d_input, num_chars * sizeof(uint8_t));
  cudaMalloc(&d_output, num_chars * sizeof(uint8_t));

  // Copy data to device
  cudaMemcpy(d_input, char_values.data(), num_chars * sizeof(uint8_t),
             cudaMemcpyHostToDevice);

  // Define the uppercase transformation operator using C++ source
  std::string cpp_source = R"(
    #include <cuda/std/cstdint>
    extern "C" __device__ void op(void* input, void* output) {
      cuda::std::uint8_t* in = (cuda::std::uint8_t*)input;
      cuda::std::uint8_t* out = (cuda::std::uint8_t*)output;
      *out = ((*in >= 97) && (*in <= 122)) ? (*in - 32) : *in;
    }
  )";

  // Create CCCL operator with C++ source
  cccl_op_t op;
  op.type = cccl_op_kind_t::CCCL_STATELESS;
  op.name = "op";
  op.code = cpp_source.c_str();
  op.code_size = cpp_source.size();
  op.code_type = CCCL_OP_CPP_SOURCE;
  op.size = 1;
  op.alignment = 1;
  op.state = nullptr;

  // Create CCCL iterators
  cccl_iterator_t input_iter =
      make_device_pointer_iterator(d_input, cccl_type_enum::CCCL_UINT8);
  cccl_iterator_t output_iter =
      make_device_pointer_iterator(d_output, cccl_type_enum::CCCL_UINT8);

  // Add memory reads/writes to state
  state.add_element_count(num_chars);
  state.add_global_memory_reads<uint8_t>(num_chars);
  state.add_global_memory_writes<uint8_t>(num_chars);

  // Build the transform
  cccl_device_transform_build_result_t build_result;
  int cc_major, cc_minor;
  cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, 0);
  cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, 0);

  // Get include paths from our CCCL installation
  std::string cub_path = "-I" + std::string(CCCL_INCLUDE_PATH) + "/cub";
  std::string thrust_path = "-I" + std::string(CCCL_INCLUDE_PATH) + "/thrust";
  std::string libcudacxx_path =
      "-I" + std::string(CCCL_INCLUDE_PATH) + "/libcudacxx/include";
  std::string ctk_path = "-I" + std::string(CTK_INCLUDE_PATH);

  CUresult build_res = cccl_device_unary_transform_build(
      &build_result, input_iter, output_iter, op, cc_major, cc_minor,
      cub_path.c_str(), thrust_path.c_str(), libcudacxx_path.c_str(),
      ctk_path.c_str());

  if (build_res != CUDA_SUCCESS) {
    state.skip("Failed to build CCCL transform");
    cudaFree(d_input);
    cudaFree(d_output);
    return;
  }

  // Run the benchmark
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch &launch) {
    // Use the stream from nvbench (cudaStream_t is same as CUstream)
    CUstream stream = (CUstream)launch.get_stream();

    // Perform the transformation using CCCL C API
    CUresult exec_res = cccl_device_unary_transform(
        build_result, input_iter, output_iter, num_chars, op, stream);

    if (exec_res != CUDA_SUCCESS) {
      // Handle error silently during benchmark
    }
  });

  // Verify the output after benchmark
  std::vector<uint8_t> output_values(num_chars);
  cudaMemcpy(output_values.data(), d_output, num_chars * sizeof(uint8_t),
             cudaMemcpyDeviceToHost);

  // Check a sample of values to verify uppercase transformation
  bool verification_passed = true;
  int errors_found = 0;
  const int max_errors_to_report = 10;

  for (size_t i = 0; i < num_chars && errors_found < max_errors_to_report;
       ++i) {
    uint8_t input_val = char_values[i];
    uint8_t output_val = output_values[i];
    uint8_t expected_val = ((input_val >= 97) && (input_val <= 122))
                               ? (input_val - 32)
                               : input_val;

    if (output_val != expected_val) {
      if (errors_found == 0) {
        std::cout << "Verification errors found:" << std::endl;
      }
      std::cout << "  Index " << i << ": input=" << static_cast<char>(input_val)
                << " (" << static_cast<int>(input_val)
                << "), output=" << static_cast<char>(output_val) << " ("
                << static_cast<int>(output_val)
                << "), expected=" << static_cast<char>(expected_val) << " ("
                << static_cast<int>(expected_val) << ")" << std::endl;
      verification_passed = false;
      errors_found++;
    }
  }

  if (verification_passed) {
    std::cout << "✓ Verification PASSED: All " << num_chars
              << " characters correctly transformed to uppercase" << std::endl;
  } else {
    std::cout << "✗ Verification FAILED: Found transformation errors"
              << std::endl;
    if (errors_found >= max_errors_to_report) {
      std::cout << "  (showing first " << max_errors_to_report
                << " errors only)" << std::endl;
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

  // Clean up
  cccl_device_transform_cleanup(&build_result);
  cudaFree(d_input);
  cudaFree(d_output);
}

// Register the benchmark with Elements axis
NVBENCH_BENCH(bench_cccl_transform_uppercase)
  .add_int64_power_of_two_axis("Elements", {20, 24, 28});

// Main function for nvbench
NVBENCH_MAIN