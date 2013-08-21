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
        /// Our OpenCL device
        cl::Device device;
        /// Our OpenCL context
        cl::Context context;
        /// Our OpenCL Command Queue
        cl::CommandQueue queue;
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

        void knnKernel(cl::Buffer& src, cl::Buffer& target, const size_t& points, const size_t& dimensions, cl::Buffer& indices, cl::Buffer& distances, const uint& k, const float& epsilon, const size_t& sourceOffset, const size_t& targetOffset, const size_t& srcSize);
    public:
        Isomap();
        ~Isomap();
        std::tuple<arma::umat, arma::fmat> knn(const arma::fmat& input, int k, float epsilon = std::numeric_limits<float>::max());
        arma::fmat embed(const arma::fmat& matrix);
    };
}

