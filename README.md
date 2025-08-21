```bash
mkdir build
cd build
cmake -GNinja -DCMAKE_BUILD_TYPE=Release ..
ninja
python3 ../generate_large_csv.py
./bench_cudf --stopping-criterion entropy
./bench_cccl --stopping-criterion entropy
```