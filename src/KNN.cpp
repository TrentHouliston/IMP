#include "Isomap.h"

#include <algorithm>
#include <string>
#include <vector>
#include <sstream>
#include <system_error>
#include <unistd.h>


namespace isomap {

    std::tuple<arma::umat, arma::fmat> Isomap::knn(const arma::fmat& input, int k, float epsilon) {

        arma::umat indices(input.n_rows, k);
        arma::fmat distances(input.n_rows, k);
        distances.fill(epsilon);

        // Check if we need to chunk (we can't allocate our distances buffer all at once
        bool chunking = gpuDevice.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() < input.n_rows * input.n_rows * sizeof(cl_float);

        // If we are not chunking
        if(!chunking) {
            
            cl::Buffer distanceMatrixBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * input.n_rows * input.n_rows);
            gpuQueue.enqueueFillBuffer(distanceMatrixBuffer, 0, 0, input.n_rows * input.n_rows * sizeof(cl_float));

            cl::Buffer colBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * input.n_rows);

            for(int i = 0; i < input.n_cols; ++i) {
                gpuQueue.enqueueWriteBuffer(colBuffer, CL_TRUE, 0, sizeof(cl_float) * input.n_rows, input.colptr(i));

                executeKernel(CL_DEVICE_TYPE_GPU, "sumColumn", input.n_rows * input.n_rows, distanceMatrixBuffer, colBuffer, cl_ulong(input.n_cols), cl_ulong(0)).wait();
            }

            executeKernel(CL_DEVICE_TYPE_GPU, "squareRoot", input.n_rows * input.n_rows, distanceMatrixBuffer).wait();

            arma::fmat distanceMatrix(input.n_rows, input.n_rows);
            gpuQueue.enqueueReadBuffer(distanceMatrixBuffer, CL_TRUE, 0, sizeof(cl_float) * input.n_rows * input.n_rows, distanceMatrix.memptr());

            // Here we map a cpu host pointer (since we are using the cpu)
            cl::Buffer cpuDistMatrixBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_float) * input.n_rows * input.n_rows, distanceMatrix.memptr());
            cl::Buffer indicesBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(cl_uint) * input.n_rows * input.n_rows, indices.memptr());
            cl::Buffer distancesBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(cl_float) * input.n_rows * input.n_rows, distances.memptr());

            // Map our buffers to opencl so the CPU opencl can search them
            gpuQueue.enqueueMapBuffer(cpuDistMatrixBuffer, CL_TRUE, CL_MAP_WRITE, 0, sizeof(cl_float) * input.n_rows * input.n_rows);
            gpuQueue.enqueueMapBuffer(indicesBuffer, CL_TRUE, CL_MAP_WRITE, 0, sizeof(cl_uint) * input.n_rows);
            gpuQueue.enqueueMapBuffer(distancesBuffer, CL_TRUE, CL_MAP_WRITE, 0, sizeof(cl_float) * input.n_rows);

            // Find our k Nearest elements
            executeKernel(CL_DEVICE_TYPE_GPU, "findKNearest", input.n_rows, cpuDistMatrixBuffer, indicesBuffer, distancesBuffer, cl_ulong(k)).wait();

            // Make sure all our data is back on the device
            gpuQueue.enqueueMapBuffer(indicesBuffer, CL_TRUE, CL_MAP_READ, 0, sizeof(cl_uint) * input.n_rows);
            gpuQueue.enqueueMapBuffer(distancesBuffer, CL_TRUE, CL_MAP_READ, 0, sizeof(cl_float) * input.n_rows);
        }
        else {
            // Check if we need to go into Superchunk mode (we can't even fit a single column of data in an allocation
            bool superchunk = gpuDevice.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() > sizeof(cl_float) * input.n_rows;

            if(!superchunk) {

                // Work out a good chunk size (that will fit the entireity of a row)
                size_t chunkSize = gpuDevice.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() / (sizeof(cl_float) * input.n_rows);
                size_t numChunks = (input.n_rows * input.n_rows) / chunkSize; // TODO if it does not evenly divide, add one

                // Allocate and fill our buffer
                cl::Buffer distanceMatrixBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * chunkSize);
                gpuQueue.enqueueFillBuffer(distanceMatrixBuffer, 0, 0, sizeof(cl_float) * chunkSize);

                for (size_t i = 0; i < numChunks; ++i) {
                    // Fill the distanceMatrixBuffer with 0s

                    for (size_t j = 0; j < input.n_cols; ++j) {
                        // Load and sum the column into the distanceMatrixBuffer

                        // Execute the kernel
                    }

                    // Save the distanceMatrixBuffer back into our in memory buffer
                }

                // Do the same as the unchunked version (find K Nearest)
                


                // Allocate a section of our total distance matrix buffer (a whole number of rows)
                // Populate this section of the distance matrix
                // Pass the distance matrix to the CPU for finding kmax

                // Allocate the next section of our total distance matrix buffer
                // Populate this section of the distance matrix
                // Pass this new distance matrix to the CPU for finding kmax
            }
            else {
                // Damn this is going to be painful... How much data did you want!
            }
        }


        return { indices, distances };

    }
}
