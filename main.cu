#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <vector>
#include <cuda_runtime.h>

int main() {
  auto stream = rmm::cuda_stream_view{};
  auto mr = cudf::get_current_device_resource_ref();

  try {
    // Read CSV file on host
    std::ifstream csv_file("lorem_ipsum.csv");
    if (!csv_file.is_open()) {
      std::cerr << "Could not open lorem_ipsum.csv" << std::endl;
      return 1;
    }
    
    // Vector to store U8 values directly
    std::vector<uint8_t> char_values;
    
    // Skip header line
    std::string header;
    std::getline(csv_file, header);
    
    // Read each line from CSV and extract the character
    std::string line;
    while (std::getline(csv_file, line)) {
      if (!line.empty()) {
        // Each line contains a single character
        char_values.push_back(static_cast<uint8_t>(line[0]));
      }
    }
    csv_file.close();
    
    std::cout << "Loaded " << char_values.size() << " characters from CSV" << std::endl;
    
    // Create a cuDF column with one U8 per row directly from the loaded data
    auto column = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::UINT8},
      char_values.size(),
      cudf::mask_state::UNALLOCATED,
      stream,
      mr
    );
    
    // Copy data from host to device
    cudaMemcpy(column->mutable_view().data<uint8_t>(), 
               char_values.data(), 
               char_values.size() * sizeof(uint8_t), 
               cudaMemcpyHostToDevice);
    
    // Print information about the column
    std::cout << "Created UINT8 column with " << column->size() << " rows" << std::endl;
    std::cout << "Column type: " << static_cast<int>(column->type().id()) << " (UINT8 = " << static_cast<int>(cudf::type_id::UINT8) << ")" << std::endl;
    
    // Extract and print first 20 characters as a string
    auto num_to_print = std::min(static_cast<cudf::size_type>(20), column->size());
    std::vector<uint8_t> first_chars(num_to_print);
    cudaMemcpy(first_chars.data(), 
               column->view().data<uint8_t>(), 
               num_to_print * sizeof(uint8_t), 
               cudaMemcpyDeviceToHost);
    
    std::cout << "\nOriginal first 20 characters: ";
    for (uint8_t ch : first_chars) {
      std::cout << static_cast<char>(ch);
    }
    std::cout << std::endl;
    
    // Use cudf::transform to convert to uppercase
    // Transform expression: if char is between 'a' (97) and 'z' (122), subtract 32
    // Otherwise keep the same value
    std::string transform_expr = 
      "__device__ inline void f(uint8_t* output, uint8_t input) { "
      "  *output = ((input >= 97) && (input <= 122)) ? (input - 32) : input; "
      "}";
    
    std::vector<cudf::column_view> input_columns = {column->view()};
    
    auto uppercase_column = cudf::transform(
      input_columns,
      transform_expr,
      cudf::data_type{cudf::type_id::UINT8},
      false,  // is_ptx = false (CUDA code)
      std::nullopt,  // no user data
      stream,
      mr
    );
    
    std::cout << "\nTransformed to uppercase using cudf::transform" << std::endl;
    std::cout << "Uppercase column has " << uppercase_column->size() << " rows" << std::endl;
    
    // Extract and print first 20 uppercase characters
    std::vector<uint8_t> upper_chars(num_to_print);
    cudaMemcpy(upper_chars.data(), 
               uppercase_column->view().data<uint8_t>(), 
               num_to_print * sizeof(uint8_t), 
               cudaMemcpyDeviceToHost);
    
    std::cout << "Uppercase first 20 characters: ";
    for (uint8_t ch : upper_chars) {
      std::cout << static_cast<char>(ch);
    }
    std::cout << std::endl;
    
    std::cout << "Uppercase UINT8 values: ";
    for (uint8_t ch : upper_chars) {
      std::cout << static_cast<int>(ch) << " ";
    }
    std::cout << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
