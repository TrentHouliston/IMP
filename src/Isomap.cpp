#include "Isomap.h"

#include <algorithm>
#include <string>
#include <vector>
#include <sstream>
#include <system_error>
#include <unistd.h>


namespace isomap {
    std::string Isomap::readKernel(const char* path) {

        char cCurrentPath[FILENAME_MAX];

        std::cout << getcwd(cCurrentPath, sizeof(cCurrentPath)) << std::endl;

        std::ifstream in(path, std::ios::in | std::ios::binary);
        if (in)
        {
            std::string contents;
            in.seekg(0, std::ios::end);
            contents.resize(in.tellg());
            in.seekg(0, std::ios::beg);
            in.read(&contents[0], contents.size());
            in.close();
            return contents;
        }
        throw(std::system_error(errno, std::system_category(), "We could not read the kernel file"));
    };

    Isomap::Isomap() {

        // Setting up our OpenCL context

        // Get all of our available platforms
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if(platforms.empty()) {
            throw std::runtime_error("No OpenCL Platforms were found, Check OpenCL installation");
        }

        // Choose our default platform
        platform = platforms.front();

        // Tell the user what OpenCL platform we are using
        std::cout << "Using OpenCL platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

        // Try to get a GPU device to work with
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if(devices.empty()) {
            std::cout << "We were unable to get a GPU device, falling back to Other OpenCL devices (this will be slow)" << std::endl;

            devices.clear();
            platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

            // If we don't have any of these then we can't continue
            std::cout << "There were no other OpenCL devices to use, Check your OpenCL installation" << std::endl;
        }

        // Pick the first device
        cl::Device device = devices.front();

        // Tell the user what Device we are using
        std::cout<< "Using OpenCL device:      " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        std::cout << "\tLargest Single Buffer: " << device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() / 1048576 << "mb" << std::endl;
        std::cout << "\tGlobal Memory Size:    " << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / 1048576 << "mb" << std::endl;

        // Load our context
        context = cl::Context(std::vector<cl::Device>{device});

        // Create our command queue
        queue = cl::CommandQueue(context, device);

        // Get our program sources
        cl::Program::Sources sources;

        // Load our KNN kernel
        std::string knnKernel = readKernel("OpenCL/KNN.cl");

        sources.push_back({knnKernel.c_str(), knnKernel.size()});

        program = cl::Program(context, sources);
        if(program.build({device}, "-cl-opt-disable -g") != CL_SUCCESS) {
            std::stringstream message;
            message << "There was an error building the OpenCL code" << std::endl
            << "Build Log:" << std::endl
            << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;

            std::cout << message.str() << std::endl;
            throw std::runtime_error(message.str());
        }
    }

    void Isomap::knnKernel(cl::Buffer& src, cl::Buffer& target, const size_t& points, const size_t& dimensions, cl::Buffer& indices, cl::Buffer& distances, const uint& k, const float& epsilon, const size_t& sourceOffset, const size_t& targetOffset, const size_t& srcSize) {

        // Our KNN kernel
        cl::Kernel kernel(program, "knn");

        kernel.setArg(0, src);     // Source matrix
        kernel.setArg(1, target);     // Target matrix (same as source since we are not chunking)
        kernel.setArg(2, cl_uint(points));           // The number of rows in the target matrix
        kernel.setArg(3, cl_uint(dimensions));       // The number of dimensions
        kernel.setArg(4, indices);    // Our Indicies output matrix
        kernel.setArg(5, distances);  // Our Distances output matrix
        kernel.setArg(6, cl_uint(k));                // The number of nearest neighbours
        kernel.setArg(7, cl_float(epsilon));          // Our epsilon limit
        kernel.setArg(8, cl_uint(0));                // Our source offset (0 since we are not chunking)
        kernel.setArg(9, cl_uint(0));                // Our target offset (0 since we are not chunking)

        // Run the kernel
        cl::Event finished;
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(srcSize), cl::NullRange, nullptr, &finished);
        finished.wait();

    }

    std::tuple<arma::umat, arma::fmat> Isomap::knn(const arma::fmat& input, int k, float epsilon) {

        bool chunking = false;

        // Armadillo uses column major ordering, the matrix is expected to be transposed
        size_t points = input.n_cols;
        size_t dimensions = input.n_rows;

        // Our output is a matrix of distances, and the corresponding indexes (transposed)
        arma::fmat distances(k, points);
        distances.fill(epsilon);
        arma::umat indices(k, points);

        // Check if we need to activate chunking mode (not enough space on the GPU for the operation
        // Check if we can allocate enough memory for our input data (in one go)
        chunking |= points * dimensions * sizeof(cl_float) > device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
        // Check if we can allocate enough memory for our index data (in one go)
        chunking |= points * k * sizeof(cl_uint) > device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
        // Check if we can allocate enough memory for our distance data (in one go)
        chunking |= points * k * sizeof(cl_float) > device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
        // Check if we have enough ram to hold it all at once
        chunking |= (points * k * (sizeof(cl_float) + sizeof(cl_uint))) + points * dimensions * sizeof(cl_float) > device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();

        chunking = true;
        // If we don't need to chunk (it will all fit on the gpu)
        if(!chunking) {
            // Allocate our buffers
            cl::Buffer sourceBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * points * dimensions);
            cl::Buffer indicesBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint) * points * k);
            cl::Buffer distancesBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * points * k);

            // Copy our source to the buffer
            queue.enqueueWriteBuffer(sourceBuffer, CL_TRUE, 0, sizeof(cl_float) * points * dimensions, input.memptr());
            // Copy our distances to the buffer (no need to write indexes)
            queue.enqueueWriteBuffer(distancesBuffer, CL_TRUE, 0, sizeof(cl_float) * points * k, distances.memptr());

            // Run the kernel
            knnKernel(sourceBuffer, sourceBuffer, points, dimensions, indicesBuffer, distancesBuffer, k, epsilon, 0, 0, points);

            // Copy our data out of the gpu and into our result
            queue.enqueueReadBuffer(distancesBuffer, CL_TRUE, 0, sizeof(cl_float) * points * k, distances.memptr());
            queue.enqueueReadBuffer(indicesBuffer, CL_TRUE, 0, sizeof(cl_uint) * points * k, indices.memptr());
        }
        else {
            // Find which of the 3 buffers takes up the largest allocation and find it's limit
            size_t numChunks = 0;
            size_t chunkSize;

            size_t minDistanceAlloc((k * sizeof(cl_float)) / device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>());
            size_t minIndexAlloc((k * sizeof(cl_uint)) / device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>());
            size_t minInputAlloc((dimensions * sizeof(cl_float)) / device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>());
            size_t minOnChip((dimensions * sizeof(cl_float) + k * (sizeof(cl_float) + sizeof(cl_uint))) / device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>());

            numChunks = minDistanceAlloc > numChunks ? minDistanceAlloc : numChunks;
            numChunks = minIndexAlloc    > numChunks ? minIndexAlloc    : numChunks;
            numChunks = minInputAlloc    > numChunks ? minInputAlloc    : numChunks;
            numChunks = minOnChip        > numChunks ? minOnChip        : numChunks;

            // We have to add one more chunk (integer math rounds down)
            numChunks++;

            numChunks = 20;

            // Work out how big each chunk is
            chunkSize = points / numChunks;

            // Work out if we need another chunk (odd number of points)
            chunkSize = chunkSize * numChunks >= points ? chunkSize : chunkSize + 1;

            // Allocate our buffers
            cl::Buffer sourceInputBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * chunkSize * dimensions);
            cl::Buffer sourceIndicesBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint) * chunkSize * k);
            cl::Buffer sourceDistancesBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * chunkSize * k);
            cl::Buffer targetInputBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * chunkSize * dimensions);
            cl::Buffer targetIndicesBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint) * chunkSize * k);
            cl::Buffer targetDistancesBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * chunkSize * k);

            for (int i = 0; i < numChunks; ++i) {

                // Work out how much data we actually have
                size_t sourceSize = chunkSize < points - i ? chunkSize : points - i;

                // Copy our data into these buffers
                queue.enqueueWriteBuffer(sourceInputBuffer, CL_TRUE, 0, sizeof(cl_float) * sourceSize * dimensions, input.colptr(i * chunkSize));
                queue.enqueueWriteBuffer(sourceDistancesBuffer, CL_TRUE, 0, sizeof(cl_float) * sourceSize * k, distances.colptr(i * chunkSize));

                for (int j = i; j < numChunks; ++j) {
                    // Find out how many datapoints we have
                    size_t targetSize = chunkSize < points - j ? chunkSize : points - j;

                    if(i != j) {
                        // Copy our data to the buffer
                        queue.enqueueWriteBuffer(targetInputBuffer, CL_TRUE, 0, sizeof(cl_float) * targetSize * dimensions, input.colptr(j * chunkSize));
                        queue.enqueueWriteBuffer(targetDistancesBuffer, CL_TRUE, 0, sizeof(cl_float) * targetSize * k, distances.colptr(j * chunkSize));

                        knnKernel(sourceInputBuffer,
                                  targetInputBuffer,
                                  targetSize,
                                  dimensions,
                                  sourceIndicesBuffer,
                                  sourceDistancesBuffer,
                                  k,
                                  epsilon,
                                  i * chunkSize,
                                  j * chunkSize,
                                  sourceSize);

                        queue.enqueueReadBuffer(sourceIndicesBuffer, CL_TRUE, 0, sizeof(cl_uint) * sourceSize * k, indices.colptr(i * chunkSize));
                        queue.enqueueReadBuffer(sourceDistancesBuffer, CL_TRUE, 0, sizeof(cl_float) * sourceSize * k, distances.colptr(i * chunkSize));

                        // Swap our buffers, so we can leave them on the GPU in the meantime (save transfer)
                        knnKernel(targetInputBuffer,
                                  sourceInputBuffer,
                                  sourceSize,
                                  dimensions,
                                  targetIndicesBuffer,
                                  targetDistancesBuffer,
                                  k,
                                  epsilon,
                                  j * chunkSize,
                                  i * chunkSize,
                                  targetSize);

                        queue.enqueueReadBuffer(targetIndicesBuffer, CL_TRUE, 0, sizeof(cl_uint) * targetSize * k, indices.colptr(j * chunkSize));
                        queue.enqueueReadBuffer(targetDistancesBuffer, CL_TRUE, 0, sizeof(cl_float) * targetSize * k, distances.colptr(j * chunkSize));
                    }
                    else {
                        // If we are running against ourself, then we can use the same buffer twice
                        knnKernel(sourceInputBuffer,
                                  sourceInputBuffer,
                                  chunkSize,
                                  dimensions,
                                  sourceIndicesBuffer,
                                  sourceDistancesBuffer,
                                  k,
                                  epsilon,
                                  i * chunkSize,
                                  i * chunkSize,
                                  sourceSize);

                        // Copy our data out and back from the OpenCL device
                        queue.enqueueReadBuffer(sourceIndicesBuffer, CL_TRUE, 0, sizeof(cl_uint), indices.colptr(i * chunkSize));
                        queue.enqueueReadBuffer(sourceDistancesBuffer, CL_TRUE, 0, sizeof(cl_float), distances.colptr(i * chunkSize));
                    }
                    
                }
            }
        }

        return { indices, distances };

    }

    Isomap::~Isomap() {
    }
}
