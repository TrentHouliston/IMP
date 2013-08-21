// K nearest Neighbours

/**
 * This kernel calculates the K nearest neighbours for the input matrix by chunking it into individual sections.
 * This allows the GPU to do a KNN on datasets that are larger then the amount of memory on the device. It is to
 * be called by splitting the input data into "Chunks" that fit on the gpu memory, and alternate them until every
 * chunk has been loaded as both source and dest. This results in a complete KNN map for all of the data.
 *
 * @param source        Our "Source" chunk of data to do KNN on
 * @param target        Our "Target" chunk, the chunk we are comparing distances to
 * @param targetSize    Our "Target" chunk size (the number of rows)
 * @param indices       The indexes to the nearest k neighbours (matches with distances)
 * @param distances     The distances to our nearest k neighbours (matches with indices)
 * @param kMax          The number of nodes to include in our K nearest
 * @param epsilon       Our epsilon to limit the distance of KMeans by
 * @param sourceOffset  The offset from 0 that the real (non chunked) index of our source is
 * @param targetOffset  The offset from 0 that the real (non chunked) index of our target is
 *
 * @author Josiah Walker
 * @author Trent Houliston
 */
kernel void knn(global const float* source,
                global const float* target,
                const unsigned int targetSize,
                const unsigned int dimensions,
                global unsigned int* indices,
                global float* distances,
                const unsigned int kMax,
                const float epsilon,
                const unsigned int sourceOffset,
                const unsigned int targetOffset) {

    // Get where we are working
    const size_t sourceBegin = get_global_id(0) * dimensions;
    const size_t sourceAddress = get_global_id(0) * kMax;

    // Check the distance for all of the points in our target chunk
    for (unsigned int i = 0; i < targetSize; i++) {

        // Calculating the distance from our source point to our target point
        float distance = 0.0;
        for (unsigned int j = 0; j < dimensions; j++) {
            
            float value = target[i * dimensions + j] - source[sourceBegin + j];
            distance += value * value;
        }
        distance = sqrt(distance);


        // Check if we are one of the K nearest (the nearest neighbours are sorted)
        if (distance <= distances[sourceAddress + kMax - 1]
            // Check we are not greater then epsilon
            && distance < epsilon
            // Check we are not looking at ourself
            && (i + targetOffset) != (get_global_id(0) + sourceOffset)) {

            // Find where we fit in in the list of nearests
            unsigned int j = 0;
            while (distance > distances[sourceAddress + j]) {
                j++;
            }

            // Shuffle all our data along to make room
            for (unsigned int k = kMax - 1; k > j; k--) {
                distances[sourceAddress + k] = distances[sourceAddress + k - 1];
                indices[sourceAddress + k] = indices[sourceAddress + k - 1];
            }

            // Add our new nearest point into it's rightful place
            distances[sourceAddress + j] = distance;
            indices[sourceAddress + j] = targetOffset + i;
        }
    }
}