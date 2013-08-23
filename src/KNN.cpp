#include "Isomap.h"

#include <algorithm>
#include <string>
#include <vector>
#include <sstream>
#include <system_error>
#include <unistd.h>


namespace isomap {

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
        chunking |= points * dimensions * sizeof(cl_float) > gpuDevice.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
        // Check if we can allocate enough memory for our index data (in one go)
        chunking |= points * k * sizeof(cl_uint) > gpuDevice.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
        // Check if we can allocate enough memory for our distance data (in one go)
        chunking |= points * k * sizeof(cl_float) > gpuDevice.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
        // Check if we have enough ram to hold it all at once
        chunking |= (points * k * (sizeof(cl_float) + sizeof(cl_uint))) + points * dimensions * sizeof(cl_float) > gpuDevice.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();

        chunking = true;
        // If we don't need to chunk (it will all fit on the gpu)
        if(!chunking) {
            // Allocate our buffers
            cl::Buffer sourceBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * points * dimensions);
            cl::Buffer indicesBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint) * points * k);
            cl::Buffer distancesBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * points * k);

            // Copy our source to the buffer
            gpuQueue.enqueueWriteBuffer(sourceBuffer, CL_TRUE, 0, sizeof(cl_float) * points * dimensions, input.memptr());
            // Copy our distances to the buffer (no need to write indexes)
            gpuQueue.enqueueWriteBuffer(distancesBuffer, CL_TRUE, 0, sizeof(cl_float) * points * k, distances.memptr());

            // Run the kernel
            executeKernel(CL_DEVICE_TYPE_GPU, "knn", points, sourceBuffer, sourceBuffer, points, dimensions, indicesBuffer, distancesBuffer, k, epsilon, 0, 0);

            // Copy our data out of the gpu and into our result
            gpuQueue.enqueueReadBuffer(distancesBuffer, CL_TRUE, 0, sizeof(cl_float) * points * k, distances.memptr());
            gpuQueue.enqueueReadBuffer(indicesBuffer, CL_TRUE, 0, sizeof(cl_uint) * points * k, indices.memptr());
        }
        else {
            // Find which of the 3 buffers takes up the largest allocation and find it's limit
            size_t numChunks = 0;
            size_t chunkSize;

            size_t minDistanceAlloc((k * sizeof(cl_float)) / gpuDevice.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>());
            size_t minIndexAlloc((k * sizeof(cl_uint)) / gpuDevice.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>());
            size_t minInputAlloc((dimensions * sizeof(cl_float)) / gpuDevice.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>());
            size_t minOnChip((dimensions * sizeof(cl_float) + k * (sizeof(cl_float) + sizeof(cl_uint))) / gpuDevice.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>());

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
                gpuQueue.enqueueWriteBuffer(sourceInputBuffer, CL_TRUE, 0, sizeof(cl_float) * sourceSize * dimensions, input.colptr(i * chunkSize));
                gpuQueue.enqueueWriteBuffer(sourceDistancesBuffer, CL_TRUE, 0, sizeof(cl_float) * sourceSize * k, distances.colptr(i * chunkSize));

                for (int j = i; j < numChunks; ++j) {
                    // Find out how many datapoints we have
                    size_t targetSize = chunkSize < points - j ? chunkSize : points - j;

                    if(i != j) {
                        // Copy our data to the buffer
                        gpuQueue.enqueueWriteBuffer(targetInputBuffer, CL_TRUE, 0, sizeof(cl_float) * targetSize * dimensions, input.colptr(j * chunkSize));
                        gpuQueue.enqueueWriteBuffer(targetDistancesBuffer, CL_TRUE, 0, sizeof(cl_float) * targetSize * k, distances.colptr(j * chunkSize));

                        executeKernel(CL_DEVICE_TYPE_GPU, "knn", sourceSize,
                                      sourceInputBuffer,
                                      targetInputBuffer,
                                      targetSize,
                                      dimensions,
                                      sourceIndicesBuffer,
                                      sourceDistancesBuffer,
                                      k,
                                      epsilon,
                                      i * chunkSize,
                                      j * chunkSize);

                        gpuQueue.enqueueReadBuffer(sourceIndicesBuffer, CL_TRUE, 0, sizeof(cl_uint) * sourceSize * k, indices.colptr(i * chunkSize));
                        gpuQueue.enqueueReadBuffer(sourceDistancesBuffer, CL_TRUE, 0, sizeof(cl_float) * sourceSize * k, distances.colptr(i * chunkSize));

                        // Swap our buffers, so we can leave them on the GPU in the meantime (save transfer)
                        executeKernel(CL_DEVICE_TYPE_GPU, "knn", targetSize,
                                      targetInputBuffer,
                                      sourceInputBuffer,
                                      sourceSize,
                                      dimensions,
                                      targetIndicesBuffer,
                                      targetDistancesBuffer,
                                      k,
                                      epsilon,
                                      j * chunkSize,
                                      i * chunkSize);

                        gpuQueue.enqueueReadBuffer(targetIndicesBuffer, CL_TRUE, 0, sizeof(cl_uint) * targetSize * k, indices.colptr(j * chunkSize));
                        gpuQueue.enqueueReadBuffer(targetDistancesBuffer, CL_TRUE, 0, sizeof(cl_float) * targetSize * k, distances.colptr(j * chunkSize));
                    }
                    else {
                        // If we are running against ourself, then we can use the same buffer twice
                        executeKernel(CL_DEVICE_TYPE_GPU, "knn", sourceSize,
                                      sourceInputBuffer,
                                      sourceInputBuffer,
                                      chunkSize,
                                      dimensions,
                                      sourceIndicesBuffer,
                                      sourceDistancesBuffer,
                                      k,
                                      epsilon,
                                      i * chunkSize,
                                      i * chunkSize);

                        // Copy our data out and back from the OpenCL device
                        gpuQueue.enqueueReadBuffer(sourceIndicesBuffer, CL_TRUE, 0, sizeof(cl_uint), indices.colptr(i * chunkSize));
                        gpuQueue.enqueueReadBuffer(sourceDistancesBuffer, CL_TRUE, 0, sizeof(cl_float), distances.colptr(i * chunkSize));
                    }
                    
                }
            }
        }

        return { indices, distances };

    }
}
