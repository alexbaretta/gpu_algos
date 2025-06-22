// Copyright (c) 2025 Alessandro Baretta
// All rights reserved.

#pragma once

#include <vector>
#include <stdexcept>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

#include "common/random.hpp"

/*
A simple tensor representation as 3D array in row-major format. The dimension,
from the innermost to the outermost, are the following:
* columns
* rows
* sheets

Note on Row-major Eigen Tensors: the elements of the first (leftmost) dimension or
index are contiguous in memory.

Each sheet is an ordinary row-major matrix. Each row is an ordinary vector.
*/

template <typename T>
struct Const {
    const T value;

    template <typename... Args>
    Const(Args&&... args) : value(args...) {}

    template <typename... Args>
    Const(const Args&... args) : value(args...) {}
};

template <typename Number>
class Tensor3D {
public:

    using Tensor = Eigen::Tensor<Number, 3>;
    using Tensor_const = Eigen::Tensor<const Number, 3>;

    using Matrix = Eigen::Matrix<Number, Eigen::Dynamic, Eigen::Dynamic>;
    // using Matrix_const = Eigen::Matrix<const Number, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Vector<Number, Eigen::Dynamic>;
    // using Vector_const = Eigen::Vector<const Number, Eigen::Dynamic>;

    using Sheet = Eigen::Map<Matrix>;
    using Sheet_const = Eigen::Map<const Matrix>;
    using Row = Eigen::Map<Vector>;
    using Row_const = Eigen::Map<const Vector>;
    using Stride = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
    using Chip = Eigen::Map<Matrix, Eigen::Unaligned, Stride>;

    const long ncols_;
    const long nrows_;
    const long nsheets_;
    std::vector<Number> vector_;

    long cols() const { return ncols_; }
    long rows() const { return nrows_; }
    long sheets() const { return nsheets_; }
    Number* data() { return vector_.data(); }
    const Number* data() const { return vector_.data(); }

    // Constructor with initialization value for all elements
    Tensor3D(const long ncols, const long nrows, const long nsheets, const Number init_value = Number{})
        : ncols_(ncols), nrows_(nrows), nsheets_(nsheets), vector_((nrows * ncols * nsheets), init_value) {}

    // // Constructor that copies data from an existing pointer
    // Tensor3D(const long ncols, const long nrows, const long nsheets, const Number* source_ptr)
    //     : ncols_(ncols), nrows_(nrows), nsheets_(nsheets), vector_(source_ptr, source_ptr + (nrows * ncols * nsheets)) {}

    // // Constructor that copies data from an existing vector
    // Tensor3D(const long ncols, const long nrows, const long nsheets, const std::vector<Number>& source_vector)
    //     : ncols_(ncols), nrows_(nrows), nsheets_(nsheets), vector_(source_vector) {
    //     // Ensure the source data has the correct size
    //     if (source_vector.size() != static_cast<size_t>(nrows * ncols * nsheets)) {
    //         throw std::invalid_argument("Source data size does not match tensor dimensions");
    //     }
    // }

    Tensor3D(const Tensor3D& other) : ncols_(other.ncols_), nrows_(other.nrows_), nsheets_(other.nsheets_), vector_(other.vector_) {}
    Tensor3D(Tensor3D&& other) : ncols_(other.ncols_), nrows_(other.nrows_), nsheets_(other.nsheets_), vector_(std::move(other.vector_)) {}
    Tensor3D(const Tensor& other) :
        ncols_(other.dimension(0)),
        nrows_(other.dimension(1)),
        nsheets_(other.dimension(2)),
        vector_(other.data(), other.data()+other.size())
        {}

    long row_size()    const { return ncols_; }
    long sheet_size()  const { return nrows_ * ncols_; }
    long tensor_size() const { return nsheets_ * sheet_size(); }

    std::size_t iat(const int col, const int row, const int sheet) const {
        return sheet * sheet_size() + row * row_size() + col;
    }
    Number& operator()(const int col, const int row, const int sheet) {
        return vector_[iat(col, row, sheet)];
    }
    const Number& operator()(const int col, const int row, const int sheet) const {
        return vector_[iat(col, row, sheet)];
    }
    Number& at(const int col, const int row, const int sheet) {
        return vector_.at(iat(col, row, sheet));
    }
    const Number& at(const int col, const int row, const int sheet) const {
        return vector_.at(iat(col, row, sheet));
    }

    Sheet sheet_at(const int sheet) {
        const auto sheet_size = ncols_ * nrows_;
        Number* const start = &vector_.at(sheet * sheet_size);
        return {start, nrows_, ncols_}; // This is the constructor of Eigen::Matrix: must be nrows, ncols
    }

    Const<Sheet> sheet_at(const int sheet) const {
        const auto sheet_size = ncols_ * nrows_;
        Number* const start = &const_cast<Tensor3D*>(this)->vector_.at(sheet * sheet_size);
        return Sheet{start, nrows_, ncols_}; // This is the constructor of Eigen::Matrix: must be nrows, ncols
    }

    Row row_at(const int row, const int sheet) {
        const auto sheet_size = ncols_ * nrows_;
        const auto row_size = ncols_;
        Number* const start = &vector_.at(sheet * sheet_size + row * row_size);
        return {start, ncols_};
    }

    Const<Row> row_at(const int row, const int sheet) const {
        assert(sheet < nsheets_);
        const auto sheet_size = ncols_ * nrows_;
        const auto row_size = ncols_;
        Number* const start = &const_cast<Tensor3D*>(this)->vector_.at(sheet * sheet_size + row * row_size);
        return Row{start, ncols_};
    }

    Chip chip_at_dim1(const int row) {
        const long outer_stride = ;
        const long inner_stride = 1;
        const Stride stride{outer_stride, inner_stride};
    }

    void randomize(const int seed) {
        randomize_vector(vector_, seed);
    }

    Eigen::TensorMap<Tensor> as_eigen_tensor() {return {data(), ncols_, nrows_, nsheets_};}
    Eigen::TensorMap<Tensor_const> as_eigen_tensor() const {return {data(), ncols_, nrows_, nsheets_};}

    // Direct print method that avoids Eigen streaming issues in CUDA
    template<typename OSTREAM>
    void print(OSTREAM& os) const {
        for (long sheet = 0; sheet < nsheets_; ++sheet) {
            // os << sheet << ": [\n";
            for (long row = 0; row < nrows_; ++row) {
                os << "(" << sheet << "," << row << "): [ ";
                for (long col = 0; col < ncols_; ++col) {
                    using PrintableType = std::conditional_t<std::is_same_v<Number, __half>, float, Number>;
                    os << static_cast<PrintableType>(at(col, row, sheet)) << " ";
                }
                os << "]\n";
            }
            // os << "]\n";
        }
    }

};
