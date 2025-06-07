// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

#pragma once

#include <vector>
#include <stdexcept>

// A simple tensor representation as 3D array with dimensions
template <typename Number>
class Tensor3D {
public:
    const long rows;
    const long cols;
    const long sheets;
    std::vector<Number> data;

    // Constructor with initialization value for all elements
    Tensor3D(const long rows, const long cols, const long sheets, const Number& init_value = Number{})
        : rows(rows), cols(cols), sheets(sheets), data((rows * cols * sheets), init_value) {}

    // Constructor that copies data from an existing pointer
    Tensor3D(const long rows, const long cols, const long sheets, const Number* source_data)
        : rows(rows), cols(cols), sheets(sheets), data(source_data, source_data + (rows * cols * sheets)) {}

    // Constructor that copies data from an existing vector
    Tensor3D(const long rows, const long cols, const long sheets, const std::vector<Number>& source_data)
        : rows(rows), cols(cols), sheets(sheets), data(source_data) {
        // Ensure the source data has the correct size
        if (source_data.size() != static_cast<size_t>(rows * cols * sheets)) {
            throw std::invalid_argument("Source data size does not match tensor dimensions");
        }
    }
};
