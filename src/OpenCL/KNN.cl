kernel void sumColumn(global float* distances,
                      global const float* column,
                      const ulong dimensions,
                      const ulong offset) {

    size_t index = get_global_id(0);
    float distance = column[(index + offset) / dimensions] - column[(index + offset) % dimensions];

    distances[index] += distance * distance;
}

kernel void squareRoot(global float* values) {

    // Square root everything
    values[get_global_id(0)] = sqrt(values[get_global_id(0)]);
}

kernel void findKNearest(const global float* input,
                         global uint* indices,
                         global float* distances,
                         const ulong k) {

    size_t startMat = get_global_id(0) * get_global_size(0);
    size_t endMat = (get_global_id(0) + 1) * get_global_size(0);

    size_t startK = get_global_id(0) * k;
    size_t endK = (get_global_id(0) + 1) * k;

    size_t ourself = startMat + get_global_id(0);

    for (size_t i = startMat; i < endMat; ++i) {

        // Check if our distance smaller then the largest distance
        if(input[i] < distances[startK]
           // Check we are not looking at our own entry
           && i != ourself) {

            // Find where we fit in the collection
            size_t us = startK;
            for(size_t j = startK; j < endK; ++j) {
                us = input[i] < distances[j] ? j : us;
            }

            // Shuffle our data back
            for(size_t j = us + 1; j > startK; --j) {
                indices[j - 1] = indices[j];
                distances[j - 1] = distances[j];
            }

            // Insert ourself
            distances[us] = input[i];
            indices[us] = i;
        }
    }
}
