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
    Number* data() { return vector_.data(); }
    const Number* data() const { return vector_.data(); }

    // Constructor with initialization value for all elements
    Tensor3D(const long rows, const long cols, const long sheets, const Number init_value = Number{})
        : rows_(rows), cols_(cols), sheets_(sheets), vector_((rows * cols * sheets), init_value) {}

    // // Constructor that copies data from an existing pointer
    // Tensor3D(const long rows, const long cols, const long sheets, const Number* source_ptr)
    //     : rows_(rows), cols_(cols), sheets_(sheets), vector_(source_ptr, source_ptr + (rows * cols * sheets)) {}

    // // Constructor that copies data from an existing vector
    // Tensor3D(const long rows, const long cols, const long sheets, const std::vector<Number>& source_vector)
    //     : rows_(rows), cols_(cols), sheets_(sheets), vector_(source_vector) {
    //     // Ensure the source data has the correct size
    //     if (source_vector.size() != static_cast<size_t>(rows * cols * sheets)) {
    //         throw std::invalid_argument("Source data size does not match tensor dimensions");
    //     }
    // }

    Tensor3D(const Tensor3D& other) : rows_(other.rows_), cols_(other.cols_), sheets_(other.sheets_), vector_(other.vector_) {}
    Tensor3D(Tensor3D&& other) : rows_(other.rows_), cols_(other.cols_), sheets_(other.sheets_), vector_(std::move(other.vector_)) {}

    long tensor_size() const { return sheets_ * cols_ * rows_; }
    long sheet_size()  const { return cols_ * rows_; }
    long col_size()    const { return rows_; }

    Number& operator()(const int col, const int row, const int sheet) {
        return vector_[sheet * sheet_size() + row * col_size() + col];
    }
    const Number& operator()(const int col, const int row, const int sheet) const {
        return vector_[sheet * sheet_size() + row * col_size() + col];
    }
    Number& at(const int col, const int row, const int sheet) {
        return vector_[sheet * sheet_size() + row * col_size() + col];
    }
    const Number& at(const int col, const int row, const int sheet) const {
        return vector_[sheet * sheet_size() + row * col_size() + col];
    }

    void randomize(const int seed) {
        randomize_vector(vector_, seed);
    }

    Eigen::TensorMap<Eigen::Tensor<Number, 3>> as_eigen_tensor() {return {data(), rows_, cols_, sheets_};}
    Eigen::TensorMap<Eigen::Tensor<const Number, 3>> as_eigen_tensor() const {return {data(), rows_, cols_, sheets_};}

    // Direct print method that avoids Eigen streaming issues in CUDA
    template<typename OSTREAM>
    void print(OSTREAM& os) const {
    for (long sheet = 0; sheet < sheets_; ++sheet) {
        os << sheet << ": [\n";
        for (long row = 0; row < rows_; ++row) {
            os << "(" << sheet << "," << row << "): [\n";
            for (long col = 0; col < cols_; ++col) {
                using PrintableType = std::conditional_t<std::is_same_v<Number, __half>, float, Number>;
                os << static_cast<PrintableType>(at(row, col, sheet)) << " ";
            }
            os << "]\n";
        }
        os << "]\n";
    }

    }

};
