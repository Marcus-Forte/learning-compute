#include <iostream>

#include <boost/compute.hpp>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <thread>
#include <execution>
#include <boost/compute/event.hpp>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <future>

namespace compute = boost::compute;

struct Point2
{
    Point2() : x(0), y(0) {}
    Point2(float a, float b) : x(a), y(b) {}
    float x;
    float y;
};
BOOST_COMPUTE_ADAPT_STRUCT(Point2, Point2, (x, y))
BOOST_COMPUTE_FUNCTION(float, dist2, (Point2 pt),
                       {
                           return sin(cos(sqrt(pt.x * pt.x + pt.y * pt.y))) + sin(cos(sqrt(pt.x * pt.x + pt.y * pt.y))) * tan(cos(sqrt(pt.x * pt.x + pt.y * pt.y)));
                       });

BOOST_COMPUTE_FUNCTION(Point2, sqrt2, (Point2 pt),
                       {
                           Point2 ret;
                           ret.x = sin(cos(sin(tan(sqrt(pt.x)))));
                           ret.y = sin(cos(sin(tan(sqrt(pt.y)))));
                           return ret;
                       });

BOOST_COMPUTE_FUNCTION(Point2, sqrt2_plus, (Point2 a, Point2 b),
                       {
                           Point2 ret;
                           ret.x = a.x + b.x;
                           ret.y = a.y + b.y;
                           return ret;
                       });

std::ostream &operator<<(std::ostream &os, const Point2 &data)
{
    return os << data.x << "," << data.y;
}
int main()
{

    // get the default device
    for (const auto &device : compute::system::devices())
    {
        std::cout << "found device: " << device.name() << std::endl;
    }
    compute::device gpu = compute::system::default_device();

    // print the device's name and platform
    std::cout << "hello from " << gpu.name();
    std::cout << " (platform: " << gpu.platform().name() << ")" << std::endl;
    std::cout << " clock: " << gpu.clock_frequency() << std::endl;

    compute::context ctx(gpu);
    compute::command_queue queue(ctx, gpu, compute::command_queue::enable_profiling);

    constexpr int num_elements = 100'000'000;
    std::cout << "Mem used: " << num_elements * sizeof(Point2) / (1024 * 1024) << "MB\n";
    // generate random numbers on the host
    std::vector<Point2> host_vector(num_elements);
    std::vector<float> host_results_h(num_elements);
    std::generate(host_vector.begin(), host_vector.end(), []() -> Point2
                  { 
        float rx = (float) rand() / (RAND_MAX / 10);
        float ry = (float) rand() / (RAND_MAX / 10); 
        return {rx, ry}; });

    auto start = std::chrono::high_resolution_clock::now();
    auto total_host = std::transform_reduce(
        host_vector.begin(), host_vector.end(), Point2(),
        // Reduce
        [](Point2 a, Point2 b)
        {
            Point2 ret;
            ret.x = a.x + b.x;
            ret.y = a.y + b.y;
            return ret;
        },
        // Transform
        [](Point2 pt)
        {
            Point2 ret;
            ret.x = sin(cos(sin(tan(sqrt(pt.x)))));
            ret.y = sin(cos(sin(tan(sqrt(pt.y)))));
            return ret;
        });
    auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
    std::cout << "Cpu TR took: " << delta << " millis\n";

    // create vector on the device
    std::cout << "Alloc  gpu\n";
    compute::event start_event = queue.enqueue_marker();
    compute::vector<Point2> device_vector(num_elements, ctx);
    compute::vector<float> device_results(num_elements, ctx);

    // Copy
    std::cout << "Copying to gpu\n";

    device_vector = host_vector;
    // compute::event end_event = queue.enqueue_marker();
    start_event.wait();
    // end_event.wait();
    queue.finish();
    std::cout << "Copy to GPU took: " << start_event.duration<boost::chrono::nanoseconds>().count() << " ns\n";

    // std::cout << "Copy to GPU took: " << end_event.duration<boost::chrono::nanoseconds>().count() << " millis\n";

    while (true) // for (int k = 0; k < 99999999; ++k)
    {
        float total_device = 0.;
        Point2 reduce_result;

        auto promise = std::async(std::launch::async, [&]() -> Point2
                                  { 
        start = std::chrono::high_resolution_clock::now();
        tbb::blocked_range<Point2 *> r(host_vector.data(), host_vector.data() + num_elements);
        auto parallel_host_res = tbb::parallel_reduce(
            r, Point2(),
            [](const tbb::blocked_range<Point2 *> &r, Point2 init) -> Point2
            {
                Point2 local;
                for (auto pt : r)
                {
                    local.x += sin(cos(sin(tan(sqrt(pt.x)))));
                    local.y += sin(cos(sin(tan(sqrt(pt.x)))));
                }
                Point2 res;
                res.x = init.x + local.x;
                res.y = init.y + local.y;
                return res;
            },
            [](Point2 a, Point2 b) -> Point2
            {
                Point2 ret;
                ret.x = a.x + b.x;
                ret.y = a.y + b.y;
                return ret;
            });
        delta = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
        std::cout << "Cpu Parallel TR took: " << delta << " millis\n";
        return parallel_host_res; });

        start = std::chrono::high_resolution_clock::now();
        // compute::transform_reduce(device_vector.begin(), device_vector.end(), &total_device, dist2, compute::plus<float>(), queue);
        compute::transform_reduce(device_vector.begin(), device_vector.end(), &reduce_result, sqrt2, sqrt2_plus, queue);
        delta = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();

        std::cout << "GPU TR took: " << delta << " millis\n";

        // promise.wait();
    //    auto parallel_host_res =  promise.get();
        std::cout << "gpu == cpu == pcpu\n";
        // std::cout << reduce_result << " == " << total_host << " == " << parallel_host_res << "\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    float total_device = 0.;
    start = std::chrono::high_resolution_clock::now();
    compute::transform_reduce(device_vector.begin(), device_vector.end(), &total_device, dist2, compute::plus<float>(), queue);
    delta = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
    std::cout << "GPU Post compilation took: " << delta << " millis\n";

    std::vector<float> host_results(num_elements);
    start = std::chrono::high_resolution_clock::now();
    compute::copy(device_results.begin(), device_results.end(), host_results.begin(), queue);
    delta = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
    std::cout << "Copy from GPU took: " << delta << " millis\n";

    std::cout << "gpu : " << host_results[0] << std::endl;
    std::cout << "cpu res: " << total_host << std::endl;

    std::cin.get();
}