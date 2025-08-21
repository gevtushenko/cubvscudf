#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <fstream>
#include <nvbench/nvbench.cuh>
#include <vector>
#include <iostream>

// Functor for uppercase transformation
struct uppercase_op {
  __device__ uint8_t operator()(uint8_t c) const {
    return ((c >= 97) && (c <= 122)) ? (c - 32) : c;
  }
};

// Benchmark function for CCCL/Thrust transform uppercase conversion
void bench_cccl_transform_uppercase(nvbench::state &state) {
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

  if (num_chars == 0) {
    state.skip("No data read from CSV");
    return;
  }

  // Create thrust device vectors
  thrust::device_vector<uint8_t> d_input(char_values);
  thrust::device_vector<uint8_t> d_output(num_chars);

  // Add memory reads/writes to state
  state.add_element_count(num_chars);
  state.add_global_memory_reads<uint8_t>(num_chars);
  state.add_global_memory_writes<uint8_t>(num_chars);

  // Run the benchmark
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch &launch) {
    // Use the stream from nvbench
    cudaStream_t stream = launch.get_stream();
    
    // Perform the transformation using Thrust with CCCL backend
    thrust::transform(thrust::cuda::par.on(stream),
                      d_input.begin(), d_input.end(),
                      d_output.begin(),
                      uppercase_op());
  });
}

// Register the benchmark
NVBENCH_BENCH(bench_cccl_transform_uppercase);

// Main function for nvbench
NVBENCH_MAIN