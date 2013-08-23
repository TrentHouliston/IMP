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

        std::vector<cl::Device> cpuDevices;
        std::vector<cl::Device> gpuDevices;

        platform.getDevices(CL_DEVICE_TYPE_CPU, &cpuDevices);
        platform.getDevices(CL_DEVICE_TYPE_GPU, &gpuDevices);

        cl::Device cpuDevice = cpuDevices.front();
        cl::Device gpuDevice = gpuDevices.front();

        // Tell the user what Device we are using
        std::cout<< "Using CPU OpenCL device:  " << cpuDevice.getInfo<CL_DEVICE_NAME>() << std::endl;
        std::cout << "\tLargest Single Buffer: " << cpuDevice.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() / 1048576 << "mb" << std::endl;
        std::cout << "\tGlobal Memory Size:    " << cpuDevice.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / 1048576 << "mb" << std::endl;
        // Tell the user what Device we are using
        std::cout<< "Using GPU OpenCL device:  " << gpuDevice.getInfo<CL_DEVICE_NAME>() << std::endl;
        std::cout << "\tLargest Single Buffer: " << gpuDevice.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() / 1048576 << "mb" << std::endl;
        std::cout << "\tGlobal Memory Size:    " << gpuDevice.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / 1048576 << "mb" << std::endl;

        // Load our context
        context = cl::Context(std::vector<cl::Device>{cpuDevice, gpuDevice});

        // Create our command queue
        cpuQueue = cl::CommandQueue(context, cpuDevice);
        gpuQueue = cl::CommandQueue(context, gpuDevice);

        // Get our program sources
        cl::Program::Sources sources;

        // Load our KNN kernel
        std::string knnKernel = readKernel("/Users/trent/Code/University/Isomap/IMP/src/OpenCL/KNN.cl");

        sources.push_back({knnKernel.c_str(), knnKernel.size()});

        program = cl::Program(context, sources);
        if(program.build({gpuDevice, cpuDevice}) != CL_SUCCESS) {
            std::stringstream message;
            message << "There was an error building the OpenCL code" << std::endl
            << "Build Log CPU:" << std::endl
            << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(gpuDevice) << std::endl << std::endl
            << "Build Log GPU:" << std::endl
            << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cpuDevice) << std::endl;

            std::cout << message.str() << std::endl;
            throw std::runtime_error(message.str());
        }
    }

    Isomap::~Isomap() {
    }
}
