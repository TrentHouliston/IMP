#include <iostream>
#include <tuple>
#include <limits>
#include <armadillo>
//#define __CL_ENABLE_EXCEPTIONS
#include <OpenCL/cl.hpp>

namespace isomap {
    class Isomap {
    private:
        /// Our OpenCL platform
        cl::Platform platform;
        /// Our CPU OpenCL device
        cl::Device cpuDevice;
        /// Our GPU OpenCL device
        cl::Device gpuDevice;
        /// Our OpenCL context
        cl::Context context;
        /// Our CPU OpenCL Command Queue
        cl::CommandQueue cpuQueue;
        /// Our GPU OpenCL Command Queue
        cl::CommandQueue gpuQueue;
        /// Our OpenCL program
        cl::Program program;

        /**
         * Reads in a kernel from a file and returns the source as a string
         *
         * @param path the path to the kernel file
         *
         * @returns the kernel at the path as a string
         */
        std::string readKernel(const char* path);

        template <int Index = 0, bool MoreArgs = true>
        struct KernelArgumentSet;

        template <int Index>
        struct KernelArgumentSet<Index, true> {
            template <typename... TArgs>
            static void set(cl::Kernel& kernel, std::tuple<TArgs...> args) {
                kernel.setArg(Index, std::get<Index>(args));

                KernelArgumentSet<Index + 1, Index + 1 < sizeof...(TArgs)>::set(kernel, args);
            }
        };

        template <int Index>
        struct KernelArgumentSet<Index, false> {
            template <typename... TArgs>
            static void set(cl::Kernel&, std::tuple<TArgs...>) {
            }
        };

        template <typename... TArgs>
        cl::Event executeKernel(const int device, const char* name, size_t numThreads, TArgs... args) {

            cl::Kernel kernel(program, name);
            KernelArgumentSet<>::set(kernel, std::make_tuple(args...));
            
            cl::Event event;
            switch(device) {
                case CL_DEVICE_TYPE_CPU:
                    cpuQueue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(numThreads), cl::NullRange, nullptr, &event);
                    break;
                case CL_DEVICE_TYPE_GPU:
                    gpuQueue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(numThreads), cl::NullRange, nullptr, &event);
                    break;
            }

            return event;
        }

    public:
        Isomap();
        ~Isomap();
        std::tuple<arma::umat, arma::fmat> knn(const arma::fmat& input, int k, float epsilon = std::numeric_limits<float>::max());
        arma::fmat embed(const arma::fmat& matrix);
    };
}

