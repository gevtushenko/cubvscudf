#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/filling.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <iostream>
#include <memory>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <tuple>
#include <vector>

int main() {
  auto stream = rmm::cuda_stream_view{};
  auto mr = cudf::get_current_device_resource_ref();

  try {
    // Read CSV file using cuDF's CSV reader
    auto source_info = cudf::io::source_info("lorem_ipsum.csv");
    auto csv_options = cudf::io::csv_reader_options::builder(source_info)
                           .header(0) // First row is header
                           .build();

    auto result = cudf::io::read_csv(csv_options, stream, mr);

    if (result.tbl->num_columns() == 0) {
      return 1;
    }

    // Get the character column (only column)
    auto char_column = result.tbl->get_column(0);

    // Check if it's a strings column
    if (char_column.type().id() != cudf::type_id::STRING) {
      return 1;
    }

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}
