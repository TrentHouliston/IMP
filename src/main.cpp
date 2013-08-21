#include <iostream>
#include <armadillo>

#include "Isomap.h"

int main(int argc, const char * argv[]) {


    isomap::Isomap isomap;

    // Format is
    std::srand(0);
    arma::fmat input = arma::randn<arma::fmat>(20000, 100).t();

    auto start = std::chrono::steady_clock::now();
    auto knn = isomap.knn(input, 10);
    auto end = std::chrono::steady_clock::now();

    std::cout << double(std::chrono::nanoseconds(end - start).count()) / double(std::nano::den) << std::endl;
    
    //std::tie(knnIndex, knnDistance) = kNearestNeighbours(); // TODO parameters (k epsilon)


    // Work out how much GPU memory we have
    // We need to split input into an appropriate sized chunks
    // These chunks need to be sized that (K * n * sizeof(size_t)) + (K * n * sizeof(double)) + 2x < gpumemory

    // Allocate a matrix to hold the results of the KNN operation (K * n doubles), (k * n size_t)

    // Call the KNN kernel for each chunk of memory passing in each input and output chunk
    
    return 0;
}

