// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

#pragma once

#include <vector>
#include <stdexcept>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

#include "common/random.hpp"

// A simple tensor representation as 3D array with dimensions
template <typename Number>
class Tensor3D {
public:
    const long rows_;
    const long cols_;
    const long sheets_;
    std::vector<Number> vector_;

    long rows() const { return rows_; }
    long cols() const { return cols_; }
    long sheets() const { return sheets_; }
    Number* data() { return vector.data(); }
    Number* data() const { return vector.data(); }

    // Constructor with initialization value for all elements
    Tensor3D(const long rows, const long cols, const long sheets, const Number& init_value = Number{})
        : rows(rows), cols(cols), sheets(sheets), vector((rows * cols * sheets), init_value) {}

    // Constructor that copies data from an existing pointer
    Tensor3D(const long rows, const long cols, const long sheets, const Number* source_ptr)
        : rows(rows), cols(cols), sheets(sheets), vector(source_ptr, source_ptr + (rows * cols * sheets)) {}

    // Constructor that copies data from an existing vector
    Tensor3D(const long rows, const long cols, const long sheets, const std::vector<Number>& source_vector)
        : rows(rows), cols(cols), sheets(sheets), vector(source_vector) {
        // Ensure the source data has the correct size
        if (source_vector.size() != static_cast<size_t>(rows * cols * sheets)) {
            throw std::invalid_argument("Source data size does not match tensor dimensions");
        }
    }

    Tensor3D(const Tensor3D& other) : rows_(other.rows_), cols_(other.cols_), sheets_(other.sheets_), vector_(other.vector_) {}
    Tensor3D(Tensor3D&& other) : rows_(other.rows_), cols_(other.cols_), sheets_(other.sheets_), vector_(std::move(other.vector_)) {}

    long tensor_size() const { return sheets_ * cols_ * rows_; }
    long sheet_size()  const { return cols_ * rows_; }
    long row_size()    const { return rows_; }

    Number& operator()(const int i, const int j, const int k) {
        return vector[k * sheet_size() + j * rows_size() + i]
    }
    const Number& operator()(const int i, const int j, const int k) const {
        return vector[k * sheet_size() + j * rows_size() + i]
    }

    void randomize(const int seed) {
        randomize_vector(vector, seed);
    }

    Eigen::TensorMap<Eigen::Tensor<Number, 3>> as_eigen_tensor() {return {data(), rows_, cols_, sheets_};}
    Eigen::TensorMap<Eigen::Tensor<const Number, 3>> as_eigen_tensor() const {return {data(), rows_, cols_, sheets_};}

};
