#include "functor.hh"
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <numeric>
#include <chrono>

namespace chrono = std::chrono;

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cout << "Usage: ./compute <num elements>\n";
        exit(0);
    }
    chrono::time_point<chrono::system_clock> start;
    int delta;

    const int num_elements = atoi(argv[1]);
    assert(num_elements > 0);

    std::vector<Point2> host_data(num_elements);

    for(int i=0; i < num_elements; ++ i) {
        host_data[i].x = rand() / ((float) RAND_MAX / 10.0f );
        host_data[i].y = rand() / ((float) RAND_MAX / 10.0f );
    }

    // Copy host data
    thrust::device_vector<Point2> device_data(host_data);
    thrust::device_vector<float> device_dists(num_elements);

    // thrust::transform(device_data.begin(), device_data.end(), device_dists.begin(), dist2());
    cudaEvent_t c_start, c_stop;
    cudaEventCreate(&c_start);
    cudaEventCreate(&c_stop);
    cudaEventRecord(c_start);
    float thrust_reduce = thrust::transform_reduce(device_data.begin(), device_data.end(), dist2(), 0., float_plus());
    cudaEventRecord(c_stop);
    cudaEventSynchronize(c_stop);
    float cuda_time;
    cudaEventElapsedTime(&cuda_time, c_start, c_stop);

    std::vector<float> device_results(num_elements);
    thrust::copy_n(device_dists.begin(), num_elements, device_results.begin());

    for(int i=0 ; i < 5; ++ i) {
        std::cout << host_data[i] << " -> " << device_results[i] << std::endl;
    }

    // CPU
    start = chrono::high_resolution_clock::now();
    float cpu_reduce = std::transform_reduce(host_data.begin(), host_data.end(), 0.0f, float_plus(), dist2());
    delta = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start).count();

    std::cout << "Thrust reduce: " << thrust_reduce << "in: " << cuda_time << " millis" << std::endl;
    std::cout << "CPU reduce: " << cpu_reduce << "in: " << delta << " milis" << std::endl;

}