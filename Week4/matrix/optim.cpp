#define CL_HPP_TARGET_OPENCL_VERSION 300
#include "matrix.h"
#include <CL/opencl.hpp>
#include <thread>
#include <immintrin.h>

matrix::matrix(unsigned long rowNum, unsigned long colNum) {
    data.resize(rowNum * colNum, 0);
    rows = rowNum;
    cols = colNum;
}

matrix::matrix(unsigned long size) {
    matrix(size, 1);
}

matrix::matrix(const matrix &other) {
    data = other.data;
    rows = other.rows;
    cols = other.cols;
}

matrix &matrix::operator=(const matrix &other) {
    // Allocate new resource
    rows = other.rows;
    cols = other.cols;
    data = other.data;

    return *this;
}

matrix operator+(const matrix &first, const matrix &second) {
    if (first.rows != second.rows || first.cols != second.cols) {
        throw std::invalid_argument("cannot add ( " + std::to_string(first.rows) + " , " + std::to_string(first.cols) + " ) with ( " + std::to_string(second.rows) + " , " + std::to_string(second.cols) + " )");
    }
    else {
        matrix sum(first.rows, first.cols);
        auto worker = [&](unsigned long startPos, unsigned long endPos) { // lambda functions goated
            for (unsigned long i = startPos; i + 4 <= endPos; i += 4) {
                __m256d a, b, c;
                a = _mm256_loadu_pd(&first.data[i]);
                b = _mm256_loadu_pd(&second.data[i]);
                c = _mm256_add_pd(a, b);
                _mm256_storeu_pd(&sum.data[i], c);
            }
            for (unsigned long i = startPos + ((endPos - startPos) / 4) * 4; i < endPos; i++) {
                sum.data[i] = first.data[i] + second.data[i];
            }
        };
        unsigned long numThreads = std::thread::hardware_concurrency();
        unsigned long totalElements = first.rows * first.cols;
        unsigned long blockSize = (totalElements + numThreads - 1) / numThreads;
        std::vector<std::thread> threads;
        for (unsigned long t = 0; t < numThreads; ++t) {
            unsigned long start = t * blockSize;
            unsigned long end = std::min(start + blockSize, totalElements);
            if (start < end) {
                threads.emplace_back(worker, start, end);
            }
        }
        for (auto &t : threads) {
            t.join();
        }
        return sum;
    }
}

matrix operator-(const matrix &first, const matrix &second) {
    if (first.rows != second.rows || first.cols != second.cols) {
        throw std::invalid_argument("cannot add ( " + std::to_string(first.rows) + " , " + std::to_string(first.cols) + " ) with ( " + std::to_string(second.rows) + " , " + std::to_string(second.cols) + " )");
    }
    else {
        matrix diff(first.rows, first.cols);
        auto worker = [&](unsigned long startPos, unsigned long endPos) {
            for (unsigned long i = startPos; i + 4 <= endPos; i += 4) {
                __m256d a, b, c;
                a = _mm256_loadu_pd(&first.data[i]);
                b = _mm256_loadu_pd(&second.data[i]);
                c = _mm256_sub_pd(a, b);
                _mm256_storeu_pd(&diff.data[i], c);
            }
            for (unsigned long i = startPos + ((endPos - startPos) / 4) * 4; i < endPos; i++) {
                diff.data[i] = first.data[i] - second.data[i];
            }
        };
        unsigned long numThreads = std::thread::hardware_concurrency();
        unsigned long totalElements = first.rows * first.cols;
        unsigned long blockSize = (totalElements + numThreads - 1) / numThreads;
        std::vector<std::thread> threads;
        for (unsigned long t = 0; t < numThreads; ++t) {
            unsigned long start = t * blockSize;
            unsigned long end = std::min(start + blockSize, totalElements);
            if (start < end) {
                threads.emplace_back(worker, start, end);
            }
        }
        for (auto &t : threads) {
            t.join();
        }
        return diff;
    }
}

matrix operator*(const matrix &first, const matrix &second) {
    if (first.rows != second.rows || first.cols != second.cols) {
        throw std::invalid_argument("cannot add ( " + std::to_string(first.rows) + " , " + std::to_string(first.cols) + " ) with ( " + std::to_string(second.rows) + " , " + std::to_string(second.cols) + " )");
    }
    else {
        matrix prod(first.rows, first.cols);
        auto worker = [&](unsigned long startPos, unsigned long endPos) {
            for (unsigned long i = startPos; i + 4 <= endPos; i += 4) {
                __m256d a, b, c;
                a = _mm256_loadu_pd(&first.data[i]);
                b = _mm256_loadu_pd(&second.data[i]);
                c = _mm256_mul_pd(a, b);
                _mm256_storeu_pd(&prod.data[i], c);
            }
            for (unsigned long i = startPos + ((endPos - startPos) / 4) * 4; i < endPos; i++) {
                prod.data[i] = first.data[i] * second.data[i];
            }
        };
        unsigned long numThreads = std::thread::hardware_concurrency();
        unsigned long totalElements = first.rows * first.cols;
        unsigned long blockSize = (totalElements + numThreads - 1) / numThreads;
        std::vector<std::thread> threads;
        for (unsigned long t = 0; t < numThreads; ++t) {
            unsigned long start = t * blockSize;
            unsigned long end = std::min(start + blockSize, totalElements);
            if (start < end) {
                threads.emplace_back(worker, start, end);
            }
        }
        for (auto &t : threads) {
            t.join();
        }
        return prod;
    }
}

matrix operator/(const matrix &first, const matrix &second) {
    if (first.rows != second.rows || first.cols != second.cols) {
        throw std::invalid_argument("cannot add ( " + std::to_string(first.rows) + " , " + std::to_string(first.cols) + " ) with ( " + std::to_string(second.rows) + " , " + std::to_string(second.cols) + " )");
    }
    else {
        matrix quotient(first.rows, first.cols);
        auto worker = [&](unsigned long startPos, unsigned long endPos) {
            for (unsigned long i = startPos; i + 4 <= endPos; i += 4) {
                __m256d a, b, c;
                a = _mm256_loadu_pd(&first.data[i]);
                b = _mm256_loadu_pd(&second.data[i]);
                c = _mm256_div_pd(a, b);
                _mm256_storeu_pd(&quotient.data[i], c);
            }
            for (unsigned long i = startPos + ((endPos - startPos) / 4) * 4; i < endPos; i++) {
                quotient.data[i] = first.data[i] / second.data[i];
            }
        };
        unsigned long numThreads = std::thread::hardware_concurrency();
        unsigned long totalElements = first.rows * first.cols;
        unsigned long blockSize = (totalElements + numThreads - 1) / numThreads;
        std::vector<std::thread> threads;
        for (unsigned long t = 0; t < numThreads; ++t) {
            unsigned long start = t * blockSize;
            unsigned long end = std::min(start + blockSize, totalElements);
            if (start < end) {
                threads.emplace_back(worker, start, end);
            }
        }
        for (auto &t : threads) {
            t.join();
        }
        return quotient;
    }
}

matrix operator*(const matrix &first, const double t) {
    matrix prod(first.rows, first.cols);
    auto worker = [&](unsigned long startPos, unsigned long endPos) {
        for (unsigned long i = startPos; i + 4 <= endPos; i += 4) {
            __m256d a, b, c;
            a = _mm256_loadu_pd(&first.data[i]);
            b = _mm256_set1_pd(t);
            c = _mm256_mul_pd(a, b);
            _mm256_storeu_pd(&prod.data[i], c);
        }
        for (unsigned long i = startPos + ((endPos - startPos) / 4) * 4; i < endPos; i++) {
            prod.data[i] = first.data[i] * t;
        }
    };
    unsigned long numThreads = std::thread::hardware_concurrency();
    unsigned long totalElements = first.rows * first.cols;
    unsigned long blockSize = (totalElements + numThreads - 1) / numThreads;
    std::vector<std::thread> threads;
    for (unsigned long t = 0; t < numThreads; ++t) {
        unsigned long start = t * blockSize;
        unsigned long end = std::min(start + blockSize, totalElements);
        if (start < end) {
            threads.emplace_back(worker, start, end);
        }
    }
    for (auto &t : threads) {
        t.join();
    }
    return prod;
}

matrix operator+(const matrix &first, const double t) {
    matrix sum(first.rows, first.cols);
    auto worker = [&](unsigned long startPos, unsigned long endPos) {
        for (unsigned long i = startPos; i + 4 <= endPos; i += 4) {
            __m256d a, b, c;
            a = _mm256_loadu_pd(&first.data[i]);
            b = _mm256_set1_pd(t);
            c = _mm256_add_pd(a, b);
            _mm256_storeu_pd(&sum.data[i], c);
        }
        for (unsigned long i = startPos + ((endPos - startPos) / 4) * 4; i < endPos; i++) {
            sum.data[i] = first.data[i] + t;
        }
    };
    unsigned long numThreads = std::thread::hardware_concurrency();
    unsigned long totalElements = first.rows * first.cols;
    unsigned long blockSize = (totalElements + numThreads - 1) / numThreads;
    std::vector<std::thread> threads;
    for (unsigned long t = 0; t < numThreads; ++t) {
        unsigned long start = t * blockSize;
        unsigned long end = std::min(start + blockSize, totalElements);
        if (start < end) {
            threads.emplace_back(worker, start, end);
        }
    }
    for (auto &t : threads) {
        t.join();
    }
    return sum;
}

matrix operator-(const matrix &first, const double t) {
    matrix diff(first.rows, first.cols);
    auto worker = [&](unsigned long startPos, unsigned long endPos) {
        for (unsigned long i = startPos; i + 4 <= endPos; i += 4) {
            __m256d a, b, c;
            a = _mm256_loadu_pd(&first.data[i]);
            b = _mm256_set1_pd(t);
            c = _mm256_sub_pd(a, b);
            _mm256_storeu_pd(&diff.data[i], c);
        }
        for (unsigned long i = startPos + ((endPos - startPos) / 4) * 4; i < endPos; i++) {
            diff.data[i] = first.data[i] - t;
        }
    };
    unsigned long numThreads = std::thread::hardware_concurrency();
    unsigned long totalElements = first.rows * first.cols;
    unsigned long blockSize = (totalElements + numThreads - 1) / numThreads;
    std::vector<std::thread> threads;
    for (unsigned long t = 0; t < numThreads; ++t) {
        unsigned long start = t * blockSize;
        unsigned long end = std::min(start + blockSize, totalElements);
        if (start < end) {
            threads.emplace_back(worker, start, end);
        }
    }
    for (auto &t : threads) {
        t.join();
    }
    return diff;
}

matrix operator/(const matrix &first, const double t) {
    matrix quotient(first.rows, first.cols);
    auto worker = [&](unsigned long startPos, unsigned long endPos) {
        for (unsigned long i = startPos; i + 4 <= endPos; i += 4) {
            __m256d a, b, c;
            a = _mm256_loadu_pd(&first.data[i]);
            b = _mm256_set1_pd(t);
            c = _mm256_div_pd(a, b);
            _mm256_storeu_pd(&quotient.data[i], c);
        }
        for (unsigned long i = startPos + ((endPos - startPos) / 4) * 4; i < endPos; i++) {
            quotient.data[i] = first.data[i] / t;
        }
    };
    unsigned long numThreads = std::thread::hardware_concurrency();
    unsigned long totalElements = first.rows * first.cols;
    unsigned long blockSize = (totalElements + numThreads - 1) / numThreads;
    std::vector<std::thread> threads;
    for (unsigned long t = 0; t < numThreads; ++t) {
        unsigned long start = t * blockSize;
        unsigned long end = std::min(start + blockSize, totalElements);
        if (start < end) {
            threads.emplace_back(worker, start, end);
        }
    }
    for (auto &t : threads) {
        t.join();
    }
    return quotient;
}

matrix matmul(const matrix &first, const matrix &second) {
    pair<unsigned long, unsigned long> dim1 = first.shape();
    pair<unsigned long, unsigned long> dim2 = second.shape();
    if (dim1.second != dim2.first) {
        throw std::invalid_argument("cannot matmul ( " + std::to_string(dim1.first) + " , " + std::to_string(dim1.second) + " ) with ( " + std::to_string(dim2.first) + " , " + std::to_string(dim2.second) + " )");
    }
    else {
        pair<unsigned long, unsigned long> dim1 = first.shape();
        pair<unsigned long, unsigned long> dim2 = second.shape();
        if (dim1.second != dim2.first) {
            throw std::invalid_argument("cannot matmul ( " + std::to_string(dim1.first) + " , " + std::to_string(dim1.second) + " ) with ( " + std::to_string(dim2.first) + " , " + std::to_string(dim2.second) + " )");
        }
        else {
            matrix net(dim1.first, dim2.second);
            cl::Platform platform = cl::Platform::getDefault();
            cl::Device device = cl::Device::getDefault();
            cl::Context context(device);
            cl::CommandQueue queue(context, device);
            cl::Program::Sources sources;
            std::string kernel_code = R"(
            __kernel void net_mul(
                __global const double* A,
                __global const double* B,
                __global double* C,
                const uint cCols,
                const uint aCols)
            {
                const uint i = get_global_id(0);
                const uint j = get_global_id(1);
                double dotPdt = 0.0;
                for (int k = 0; k < aCols; k++) {
                    dotPdt += A[(i * aCols) + k] * B[(k * cCols) + j];
                }
                C[(i * cCols) + j] = dotPdt;
            }
        )";
            sources.push_back({kernel_code.c_str(), kernel_code.length()});

            cl::Program program(context, sources);
            program.build({device});

            cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * first.data.size(), (void *)first.data.data());
            cl::Buffer bufferB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * second.data.size(), (void *)second.data.data());
            cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY, sizeof(double) * net.data.size());

            cl::Kernel kernel(program, "net_mul");
            kernel.setArg(0, bufferA);
            kernel.setArg(1, bufferB);
            kernel.setArg(2, bufferC);
            kernel.setArg(3, static_cast<uint>(net.cols));
            kernel.setArg(4, static_cast<uint>(first.cols)); // Very annoying error for which I was forced to use static_cast cri :(

            cl::NDRange global(net.rows, net.cols);

            queue.enqueueNDRangeKernel(kernel, cl::NullRange, global);
            queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, sizeof(double) * net.data.size(), net.data.data());
            return net;
        }
    }
}

matrix zeros(unsigned long rows, unsigned long cols) {
    // maybe optimize
    return matrix(rows, cols);
}

matrix zeros(unsigned long size) {
    // maybe optimize
    return matrix(size);
}

matrix eye(unsigned long size) {
    matrix diag(size, size);
    for (int i = 0; i < size; i++) {
        diag(i, i) = 1;
    }
    return diag;
}

matrix eye(unsigned long rows, unsigned long cols) {
    matrix diag(rows, cols);
    for (int i = 0; i < min(rows, cols); i++) {
        diag(i, i) = 1;
    }
    return diag;
}

matrix identity(unsigned long size) {
    return eye(size);
}

void findColumnMax(matrix &arr, matrix &result, unsigned long startCol, unsigned long endCol) {
    unsigned long arrRows = arr.rows;
    for (unsigned long col = startCol; col < endCol; ++col) {
        double max_value = arr(0, col);
        for (unsigned long row = 1; row < arrRows; ++row) {
            max_value = max(max_value, arr(row, col));
        }
        result(0, col) = max_value;
    }
}

void findColumnMin(matrix &arr, matrix &result, unsigned long startCol, unsigned long endCol) {
    unsigned long arrRows = arr.rows;
    for (unsigned long col = startCol; col < endCol; ++col) {
        double min_value = arr(0, col);
        for (unsigned long row = 1; row < arrRows; ++row) {
            min_value = min(min_value, arr(row, col));
        }
        result(0, col) = min_value;
    }
}

void findRowMax(matrix &arr, matrix &result, unsigned long startRow, unsigned long endRow) {
    unsigned long arrCols = arr.cols;
    for (unsigned long row = startRow; row < endRow; ++row) {
        double max_value = arr(row, 0);
        for (unsigned long col = 1; col < arrCols; ++col) {
            max_value = max(max_value, arr(row, col));
        }
        result(row, 0) = max_value;
    }
}

void findRowMin(matrix &arr, matrix &result, unsigned long startRow, unsigned long endRow) {
    unsigned long arrCols = arr.cols;
    for (unsigned long row = startRow; row < endRow; ++row) {
        double min_value = arr(row, 0);
        for (unsigned long col = 1; col < arrCols; ++col) {
            min_value = min(min_value, arr(row, col));
        }
        result(row, 0) = min_value;
    }
}

void findArgColumnMax(matrix &arr, matrix &result, unsigned long startCol, unsigned long endCol) {
    unsigned long arrRows = arr.rows;
    for (unsigned long col = startCol; col < endCol; ++col) {
        double max_value = arr(0, col);
        unsigned long max_index = 0;
        for (unsigned long row = 1; row < arrRows; ++row) {
            if (arr(row, col) > max_value) {
                max_value = arr(row, col);
                max_index = row;
            }
        }
        result(0, col) = max_index;
    }
}

void findArgColumnMin(matrix &arr, matrix &result, unsigned long startCol, unsigned long endCol) {
    unsigned long arrRows = arr.rows;
    for (unsigned long col = startCol; col < endCol; ++col) {
        double min_value = arr(0, col);
        unsigned long min_index = 0;
        for (unsigned long row = 1; row < arrRows; ++row) {
            if (arr(row, col) < min_value) {
                min_value = arr(row, col);
                min_index = row;
            }
        }
        result(0, col) = min_index;
    }
}

void findArgRowMax(matrix &arr, matrix &result, unsigned long startRow, unsigned long endRow)
{
    unsigned long arrCols = arr.cols;
    for (unsigned long row = startRow; row < endRow; ++row) {
        double max_value = arr(row, 0);
        unsigned long max_index = 0;
        for (unsigned long col = 1; col < arrCols; ++col) {
            if (arr(row, col) > max_value) {
                max_value = arr(row, col);
                max_index = col;
            }
        }
        result(row, 0) = max_index;
    }
}

void findArgRowMin(matrix &arr, matrix &result, unsigned long startRow, unsigned long endRow) {
    unsigned long arrCols = arr.cols;
    for (unsigned long row = startRow; row < endRow; ++row) {
        double min_value = arr(row, 0);
        unsigned long min_index = 0;
        for (unsigned long col = 1; col < arrCols; ++col) {
            if (arr(row, col) < min_value) {
                min_value = arr(row, col);
                min_index = col;
            }
        }
        result(row, 0) = min_index;
    }
}

matrix max(matrix &arr, int axis) {
    if (axis < 0 || axis > 1)
        throw std::invalid_argument("Axis must be 0 or 1");

    unsigned long arrRows = arr.rows;
    unsigned long arrCols = arr.cols;

    matrix result(axis == 0 ? 1 : arrRows, axis == 0 ? arrCols : 1);

    if (axis == 0) {
        unsigned long numThreads = min(arrCols, ulong(thread::hardware_concurrency()));
        vector<thread> threads;
        unsigned long chunkSize = arrCols / numThreads;
        unsigned long startCol = 0;
        for (unsigned long i = 0; i < numThreads; ++i) {
            unsigned long endCol = (i == numThreads - 1) ? arrCols : startCol + chunkSize;
            threads.push_back(thread(findColumnMax, ref(arr), ref(result), startCol, endCol));
            startCol = endCol;
        }
        for (auto &t : threads) {
            t.join();
        }
    }
    else {
        unsigned long numThreads = min(arrRows, ulong(thread::hardware_concurrency()));
        vector<thread> threads;
        unsigned long chunkSize = arrRows / numThreads;
        unsigned long startRow = 0;
        for (unsigned long i = 0; i < numThreads; ++i) {
            unsigned long endRow = (i == numThreads - 1) ? arrRows : startRow + chunkSize;
            threads.push_back(thread(findRowMax, ref(arr), ref(result), startRow, endRow));
            startRow = endRow;
        }
        for (auto &t : threads) {
            t.join();
        }
    }

    return result;
}

matrix argmax(matrix &arr, int axis) {
    if (axis < 0 || axis > 1)
        throw std::invalid_argument("Axis must be 0 or 1");

    unsigned long arrRows = arr.rows;
    unsigned long arrCols = arr.cols;

    matrix result(axis == 0 ? 1 : arrRows, axis == 0 ? arrCols : 1);

    if (axis == 0) {
        unsigned long numThreads = min(arrCols, ulong(thread::hardware_concurrency()));
        vector<thread> threads;
        unsigned long chunkSize = arrCols / numThreads;
        unsigned long startCol = 0;
        for (unsigned long i = 0; i < numThreads; ++i) {
            unsigned long endCol = (i == numThreads - 1) ? arrCols : startCol + chunkSize;
            threads.push_back(thread(findArgColumnMax, ref(arr), ref(result), startCol, endCol));
            startCol = endCol;
        }
        for (auto &t : threads) {
            t.join();
        }
    }
    else {
        unsigned long numThreads = min(arrRows, ulong(thread::hardware_concurrency()));
        vector<thread> threads;
        unsigned long chunkSize = arrRows / numThreads;
        unsigned long startRow = 0;
        for (unsigned long i = 0; i < numThreads; ++i) {
            unsigned long endRow = (i == numThreads - 1) ? arrRows : startRow + chunkSize;
            threads.push_back(thread(findArgRowMax, ref(arr), ref(result), startRow, endRow));
            startRow = endRow;
        }
        for (auto &t : threads) {
            t.join();
        }
    }

    return result;
}

matrix max(matrix &arr) {
    unsigned long arrRows = arr.rows;
    unsigned long arrCols = arr.cols;
    if (arrRows != 1 && arrCols != 1) {
        matrix result(1, arrCols);
        unsigned long num_threads = min(arrCols, ulong(thread::hardware_concurrency()));
        vector<thread> threads;
        unsigned long chunkSize = arrCols / num_threads;
        unsigned long startCol = 0;
        for (unsigned long i = 0; i < num_threads; ++i) {
            unsigned long endCol = (i == num_threads - 1) ? arrCols : startCol + chunkSize;
            threads.push_back(thread(findColumnMax, ref(arr), ref(result), startCol, endCol));
            startCol = endCol;
        }
        for (auto &t : threads) {
            t.join();
        }
        return result;
    }
    else {
        matrix result(1, 1);
        result(0, 0) = arr(0, 0);
        for (int i = 0; i < arr.cols * arr.rows; i++) {
            result(0, 0) = max(result(0, 0), arr.data[i]);
        }
        return result;
    }
}

matrix argmax(matrix &arr) {
    unsigned long arrRows = arr.rows;
    unsigned long arrCols = arr.cols;
    if (arrRows != 1 && arrCols != 1) {
        matrix result(1, arrCols);
        unsigned long num_threads = min(arrCols, ulong(thread::hardware_concurrency()));
        vector<thread> threads;
        unsigned long chunkSize = arrCols / num_threads;
        unsigned long startCol = 0;
        for (unsigned long i = 0; i < num_threads; ++i) {
            unsigned long endCol = (i == num_threads - 1) ? arrCols : startCol + chunkSize;
            threads.push_back(thread(findArgColumnMax, ref(arr), ref(result), startCol, endCol));
            startCol = endCol;
        }
        for (auto &t : threads) {
            t.join();
        }
        return result;
    }
    else {
        matrix result(1, 1);
        result(0, 0) = arr(0, 0);
        for (int i = 0; i < arr.cols * arr.rows; i++) {
            result(0, 0) = max(result(0, 0), arr.data[i]);
        }
        return result;
    }
}

matrix min(matrix &arr, int axis) {
    if (axis < 0 || axis > 1)
        throw std::invalid_argument("Axis must be 0 or 1");

    unsigned long arrRows = arr.rows;
    unsigned long arrCols = arr.cols;

    matrix result(axis == 0 ? 1 : arrRows, axis == 0 ? arrCols : 1);

    if (axis == 0) {
        unsigned long numThreads = min(arrCols, ulong(thread::hardware_concurrency()));
        vector<thread> threads;
        unsigned long chunkSize = arrCols / numThreads;
        unsigned long startCol = 0;
        for (unsigned long i = 0; i < numThreads; ++i) {
            unsigned long endCol = (i == numThreads - 1) ? arrCols : startCol + chunkSize;
            threads.push_back(thread(findColumnMin, ref(arr), ref(result), startCol, endCol));
            startCol = endCol;
        }
        for (auto &t : threads) {
            t.join();
        }
    }
    else {
        unsigned long numThreads = min(arrRows, ulong(thread::hardware_concurrency()));
        vector<thread> threads;
        unsigned long chunkSize = arrRows / numThreads;
        unsigned long startRow = 0;
        for (unsigned long i = 0; i < numThreads; ++i) {
            unsigned long endRow = (i == numThreads - 1) ? arrRows : startRow + chunkSize;
            threads.push_back(thread(findRowMin, ref(arr), ref(result), startRow, endRow));
            startRow = endRow;
        }
        for (auto &t : threads) {
            t.join();
        }
    }

    return result;
}

matrix argmin(matrix &arr, int axis) {
    if (axis < 0 || axis > 1)
        throw std::invalid_argument("Axis must be 0 or 1");

    unsigned long arrRows = arr.rows;
    unsigned long arrCols = arr.cols;

    matrix result(axis == 0 ? 1 : arrRows, axis == 0 ? arrCols : 1);

    if (axis == 0) {
        unsigned long numThreads = min(arrCols, ulong(thread::hardware_concurrency()));
        vector<thread> threads;
        unsigned long chunkSize = arrCols / numThreads;
        unsigned long startCol = 0;
        for (unsigned long i = 0; i < numThreads; ++i) {
            unsigned long endCol = (i == numThreads - 1) ? arrCols : startCol + chunkSize;
            threads.push_back(thread(findArgColumnMin, ref(arr), ref(result), startCol, endCol));
            startCol = endCol;
        }
        for (auto &t : threads) {
            t.join();
        }
    }
    else {
        unsigned long numThreads = min(arrRows, ulong(thread::hardware_concurrency()));
        vector<thread> threads;
        unsigned long chunkSize = arrRows / numThreads;
        unsigned long startRow = 0;
        for (unsigned long i = 0; i < numThreads; ++i) {
            unsigned long endRow = (i == numThreads - 1) ? arrRows : startRow + chunkSize;
            threads.push_back(thread(findArgRowMin, ref(arr), ref(result), startRow, endRow));
            startRow = endRow;
        }
        for (auto &t : threads) {
            t.join();
        }
    }

    return result;
}

matrix min(matrix &arr) {
    unsigned long arrRows = arr.rows;
    unsigned long arrCols = arr.cols;
    if (arrRows != 1 && arrCols != 1) {
        matrix result(1, arrCols);
        unsigned long num_threads = min(arrCols, ulong(thread::hardware_concurrency()));
        vector<thread> threads;
        unsigned long chunkSize = arrCols / num_threads;
        unsigned long startCol = 0;
        for (unsigned long i = 0; i < num_threads; ++i) {
            unsigned long endCol = (i == num_threads - 1) ? arrCols : startCol + chunkSize;
            threads.push_back(thread(findColumnMin, ref(arr), ref(result), startCol, endCol));
            startCol = endCol;
        }
        for (auto &t : threads) {
            t.join();
        }
        return result;
    }
    else {
        matrix result(1, 1);
        result(0, 0) = arr(0, 0);
        for (int i = 0; i < arr.cols * arr.rows; i++) {
            result(0, 0) = min(result(0, 0), arr.data[i]);
        }
        return result;
    }
}

matrix argmin(matrix &arr) {
    unsigned long arrRows = arr.rows;
    unsigned long arrCols = arr.cols;
    if (arrRows != 1 && arrCols != 1) {
        matrix result(1, arrCols);
        unsigned long num_threads = min(arrCols, ulong(thread::hardware_concurrency()));
        vector<thread> threads;
        unsigned long chunkSize = arrCols / num_threads;
        unsigned long startCol = 0;
        for (unsigned long i = 0; i < num_threads; ++i) {
            unsigned long endCol = (i == num_threads - 1) ? arrCols : startCol + chunkSize;
            threads.push_back(thread(findArgColumnMin, ref(arr), ref(result), startCol, endCol));
            startCol = endCol;
        }
        for (auto &t : threads) {
            t.join();
        }
        return result;
    }
    else {
        matrix result(1, 1);
        result(0, 0) = arr(0, 0);
        for (int i = 0; i < arr.cols * arr.rows; i++) {
            result(0, 0) = min(result(0, 0), arr.data[i]);
        }
        return result;
    }
}

matrix ones(unsigned long rows, unsigned long cols) {
    matrix t(rows, cols);
    auto worker = [&](unsigned long startPos, unsigned long endPos) {
        for (unsigned long i = startPos; i + 4 <= endPos; i += 4) {
            __m256d a = _mm256_set1_pd(1);
            _mm256_storeu_pd(&t.data[i], a);
        }
        for (unsigned long i = startPos + ((endPos - startPos) / 4) * 4; i < endPos; i++) {
            t.data[i] = 1;
        }
    };
    unsigned long numThreads = std::thread::hardware_concurrency();
    unsigned long totalElements = rows * cols;
    unsigned long blockSize = (totalElements + numThreads - 1) / numThreads;
    std::vector<std::thread> threads;
    for (unsigned long t = 0; t < numThreads; ++t) {
        unsigned long start = t * blockSize;
        unsigned long end = std::min(start + blockSize, totalElements);
        if (start < end) {
            threads.emplace_back(worker, start, end);
        }
    }
    for (auto &t : threads) {
        t.join();
    }
    return t;
}

matrix fabs(matrix &a) {
    matrix res(a.rows, a.cols);
    cl::Platform platform = cl::Platform::getDefault();
    cl::Device device = cl::Device::getDefault();
    cl::Context context(device);
    cl::CommandQueue queue(context, device);
    cl::Program::Sources sources;
    std::string kernel_code = R"(
            __kernel void set_fabs(
                __global const double* A,
                __global double* B)
            {
                const uint n = get_global_id(0);
                B[n] = fabs(A[n]);
            }
        )";
    sources.push_back({kernel_code.c_str(), kernel_code.length()});

    cl::Program program(context, sources);
    program.build({device});

    cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * a.data.size(), (void *)a.data.data());
    cl::Buffer bufferB(context, CL_MEM_WRITE_ONLY, sizeof(double) * res.data.size());

    cl::Kernel kernel(program, "set_fabs");
    kernel.setArg(0, bufferA);
    kernel.setArg(1, bufferB);

    cl::NDRange global(res.rows * res.cols);

    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global);
    queue.enqueueReadBuffer(bufferB, CL_TRUE, 0, sizeof(double) * res.data.size(), res.data.data());
    return res;
}

matrix exp(matrix &a) {
    matrix res(a.rows, a.cols);
    cl::Platform platform = cl::Platform::getDefault();
    cl::Device device = cl::Device::getDefault();
    cl::Context context(device);
    cl::CommandQueue queue(context, device);
    cl::Program::Sources sources;
    std::string kernel_code = R"(
            __kernel void set_exp(
                __global const double* A,
                __global double* B)
            {
                const uint n = get_global_id(0);
                B[n] = exp(A[n]);
            }
        )";
    sources.push_back({kernel_code.c_str(), kernel_code.length()});

    cl::Program program(context, sources);
    program.build({device});

    cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * a.data.size(), (void *)a.data.data());
    cl::Buffer bufferB(context, CL_MEM_WRITE_ONLY, sizeof(double) * res.data.size());

    cl::Kernel kernel(program, "set_exp");
    kernel.setArg(0, bufferA);
    kernel.setArg(1, bufferB);

    cl::NDRange global(res.rows * res.cols);

    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global);
    queue.enqueueReadBuffer(bufferB, CL_TRUE, 0, sizeof(double) * res.data.size(), res.data.data());
    return res;
}

matrix tanh(matrix &a) {
    matrix res(a.rows, a.cols);
    cl::Platform platform = cl::Platform::getDefault();
    cl::Device device = cl::Device::getDefault();
    cl::Context context(device);
    cl::CommandQueue queue(context, device);
    cl::Program::Sources sources;
    std::string kernel_code = R"(
            __kernel void set_tanh(
                __global const double* A,
                __global double* B)
            {
                const uint n = get_global_id(0);
                B[n] = tanh(A[n]);
            }
        )";
    sources.push_back({kernel_code.c_str(), kernel_code.length()});

    cl::Program program(context, sources);
    program.build({device});

    cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * a.data.size(), (void *)a.data.data());
    cl::Buffer bufferB(context, CL_MEM_WRITE_ONLY, sizeof(double) * res.data.size());

    cl::Kernel kernel(program, "set_tanh");
    kernel.setArg(0, bufferA);
    kernel.setArg(1, bufferB);

    cl::NDRange global(res.rows * res.cols);

    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global);
    queue.enqueueReadBuffer(bufferB, CL_TRUE, 0, sizeof(double) * res.data.size(), res.data.data());
    return res;
}

matrix log(matrix &a, double logbase) {
    matrix res(a.rows, a.cols);
    cl::Platform platform = cl::Platform::getDefault();
    cl::Device device = cl::Device::getDefault();
    cl::Context context(device);
    cl::CommandQueue queue(context, device);
    cl::Program::Sources sources;
    std::string kernel_code = R"(
            __kernel void set_log(
                __global const double* A,
                __global double* B,
                const double logbase)
            {
                const uint n = get_global_id(0);
                B[n] = log(A[n]) / log(logbase);
            }
        )";
    sources.push_back({kernel_code.c_str(), kernel_code.length()});

    cl::Program program(context, sources);
    program.build({device});

    cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * a.data.size(), (void *)a.data.data());
    cl::Buffer bufferB(context, CL_MEM_WRITE_ONLY, sizeof(double) * res.data.size());

    cl::Kernel kernel(program, "set_log");
    kernel.setArg(0, bufferA);  
    kernel.setArg(1, bufferB);
    kernel.setArg(2, logbase);

    cl::NDRange global(res.rows * res.cols);

    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global);
    queue.enqueueReadBuffer(bufferB, CL_TRUE, 0, sizeof(double) * res.data.size(), res.data.data());
    return res;
}

matrix sqrt(matrix &a) {
    matrix res(a.rows, a.cols);
    cl::Platform platform = cl::Platform::getDefault();
    cl::Device device = cl::Device::getDefault();
    cl::Context context(device);
    cl::CommandQueue queue(context, device);
    cl::Program::Sources sources;
    std::string kernel_code = R"(
            __kernel void set_sqrt(
                __global const double* A,
                __global double* B)
            {
                const uint n = get_global_id(0);
                B[n] = native_sqrt(A[n]);
            }
        )";
    sources.push_back({kernel_code.c_str(), kernel_code.length()});

    cl::Program program(context, sources);
    program.build({device});

    cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * a.data.size(), (void *)a.data.data());
    cl::Buffer bufferB(context, CL_MEM_WRITE_ONLY, sizeof(double) * res.data.size());

    cl::Kernel kernel(program, "set_sqrt");
    kernel.setArg(0, bufferA);
    kernel.setArg(1, bufferB);

    cl::NDRange global(res.rows * res.cols);

    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global);
    queue.enqueueReadBuffer(bufferB, CL_TRUE, 0, sizeof(double) * res.data.size(), res.data.data());
    return res;
}

void partiallyEliminateColumn(matrix &a, unsigned long i, unsigned long startRow, unsigned long endRow) {
    unsigned long dim = a.shape().second;
    for (unsigned long j = startRow; j < endRow; ++j) {
        double factor = a(j, i) / a(i, i);
        __m256d factorVec = _mm256_set1_pd(factor);
        for (unsigned long k = i; k + 4 <= dim; k += 4) {
            __m256d currentRow = _mm256_loadu_pd(&a(j, k));
            __m256d pivotRow = _mm256_loadu_pd(&a(i, k));
            __m256d toBeSubtracted = _mm256_mul_pd(factorVec, pivotRow);
            __m256d result = _mm256_sub_pd(currentRow, toBeSubtracted);
            _mm256_storeu_pd(&a(j, k), result);
        }
        for (unsigned long k = (dim / 4) * 4; k < dim; ++k) {
            a(j, k) -= factor * a(i, k);
        }
    }
}

void fullyEliminateColumn(matrix &a, unsigned long i, unsigned long startRow, unsigned long endRow) {
    unsigned long dim = a.shape().second;
    for (unsigned long j = startRow; j < endRow; ++j) {
        if (i == j)
            continue;
        double factor = a(j, i) / a(i, i);
        __m256d factorVec = _mm256_set1_pd(factor);
        for (unsigned long k = 0; k + 4 <= dim; k += 4) {
            __m256d currentRow = _mm256_loadu_pd(&a(j, k));
            __m256d pivotRow = _mm256_loadu_pd(&a(i, k));
            __m256d toBeSubtracted = _mm256_mul_pd(factorVec, pivotRow);
            __m256d result = _mm256_sub_pd(currentRow, toBeSubtracted);
            _mm256_storeu_pd(&a(j, k), result);
        }
        for (unsigned long k = (dim / 4) * 4; k < dim; ++k) {
            a(j, k) -= factor * a(i, k);
        }
    }
}

void assignRow(matrix &a, unsigned long startRow, unsigned long endRow) {
    unsigned long dim = a.shape().second;
}

matrix matrix::inverse() {
    matrix a = *this;
    pair<unsigned long, unsigned long> dim = a.shape();
    if (dim.first != dim.second)
        throw std::invalid_argument("Cannot invert ( " + std::to_string(dim.first) + " , " + std::to_string(dim.second) + " )");
    unsigned long n = a.rows;
    matrix augmented(n, 2 * n);
    // Initialize the augmented matrix with the identity matrix on the right
    auto assignAugmented = [&](ulong startrow, ulong endrow) {
        for (ulong i = startrow; i < endrow; i++) {
            __m256d forStorage;
            __m256d setZero = _mm256_set1_pd(0);
            for (ulong j = 0; j + 4 <= n; j += 4) {
                forStorage = _mm256_loadu_pd(&a(i, j));
                _mm256_storeu_pd(&augmented(i, j), forStorage);
            }
            for (ulong j = (n / 4) * 4; j < n; j++) {
                augmented(i, j) = a(i, j);
            }
            augmented(i, i + n) = 1;
        }
    };

    unsigned long numThreads = thread::hardware_concurrency();
    unsigned long blockSize = (n + numThreads - 1) / numThreads;
    vector<thread> threads;
    for (unsigned long t = 0; t < numThreads; ++t) {
        unsigned long start = t * blockSize;
        unsigned long end = std::min(start + blockSize, n);
        if (start < end) {
            threads.emplace_back(assignAugmented, start, end);
        }
    }
    for (auto &t : threads) {
        t.join();
    }

    // Perform Gauss-Jordan elimination
    for (unsigned long i = 0; i < n; ++i) {
        // Find the pivot
        double pivot = augmented(i, i);
        if (pivot == 0.0) {
            throw runtime_error("Matrix is singular and cannot be inverted.");
        }

        // Normalize the pivot row
        __m256d toDivide = _mm256_set1_pd(pivot);
        __m256d row, normalized;
        for (unsigned long j = 0; j + 4 <= 2 * n; j += 4) {
            row = _mm256_loadu_pd(&augmented(i, j));
            normalized = _mm256_div_pd(row, toDivide);
            _mm256_storeu_pd(&augmented(i, j), normalized);
        }
        for (unsigned long j = ((2 * n) / 4) * 4; j < 2 * n; j++) {
            augmented(i, j) /= pivot;
        }

        // Eliminate the current column in other rows
        unsigned long chunkSize = (n + numThreads - 1) / numThreads;
        threads.clear();
        for (unsigned long t = 0; t < numThreads; ++t) {
            unsigned long start = t * chunkSize;
            unsigned long end = min(start + chunkSize, n);
            if (start < end) {
                threads.emplace_back(fullyEliminateColumn, ref(augmented), i, start, end);
            }
        }
        for (auto &t : threads) {
            t.join();
        }
    }
    // Extract the inverse matrix from the augmented matrix
    matrix result(n, n);
    auto assignRowWorker = [&](ulong startrow, ulong endrow) {
        for (ulong i = startrow; i < endrow; i++) {
            __m256d forStorage;
            for (unsigned long j = 0; j + 4 <= n; j += 4) {
                forStorage = _mm256_loadu_pd(&augmented(i, j + n));
                _mm256_storeu_pd(&result(i, j), forStorage);
            }
            for (unsigned long j = (n / 4) * 4; j < n; j++) {
                result(i, j) = augmented(i, j + n);
            }
        }
    };

    blockSize = (n + numThreads - 1) / numThreads;
    threads.clear();
    for (unsigned long t = 0; t < numThreads; ++t) {
        unsigned long start = t * blockSize;
        unsigned long end = std::min(start + blockSize, n);
        if (start < end) {
            threads.emplace_back(assignRowWorker, start, end);
        }
    }
    for (auto &t : threads) {
        t.join();
    }
    return result;
}

matrix matrix::transpose() {
    pair<unsigned long, unsigned long> dim = this->shape();
    matrix T(dim.second, dim.first);
    unsigned long numThreads = thread::hardware_concurrency();
    unsigned long blockSize = (dim.first + numThreads - 1) / numThreads;
    vector<thread> threads;

    auto transposeBlock = [&](unsigned long start, unsigned long end) {
        for (unsigned long i = start; i < end; ++i) {
            for (unsigned long j = 0; j < dim.second; ++j) {
                T(j, i) = (*this)(i, j);
            }
        }
    };

    for (unsigned long t = 0; t < numThreads; ++t) {
        unsigned long start = t * blockSize;
        unsigned long end = std::min(start + blockSize, rows);
        if (start < end) {
            threads.emplace_back(transposeBlock, start, end);
        }
    }
    for (auto &t : threads) {
        t.join();
    }
    return T;
}

double matrix::determinant() {
    if (rows != cols) {
        throw std::invalid_argument("Matrix must be square to calculate determinant");
    }
    unsigned long n = rows;
    matrix a(*this); // Make a copy of the matrix

    double det = 1;
    for (unsigned long i = 0; i < n; ++i) {
        // Find the pivot
        unsigned long pivot = i;
        for (unsigned long j = i + 1; j < n; ++j) {
            if (abs(a.data[(j * n) + i]) > abs(a.data[(pivot * n) + i])) {
                pivot = j;
            }
        }

        // Swap rows if needed
        if (pivot != i) {
            for (unsigned long k = 0; k < n; ++k) {
                std::swap(a.data[(i * n) + k], a.data[(pivot * n) + k]);
            }
            det *= -1; // Swap changes the sign of the determinant
        }

        // Check for zero pivot
        if (a.data[(i * n) + i] == 0) {
            return 0; // Determinant is zero
        }

        // Eliminate the column
        unsigned long numThreads = min(n - i - 1, ulong(thread::hardware_concurrency()));
        if (numThreads == 0) {
            numThreads = 1;
        }
        unsigned long chunkSize = (n - i - 1) / numThreads;
        vector<thread> threads;
        unsigned long startRow = i + 1;
        for (unsigned long t = 0; t < numThreads; ++t) {
            unsigned long endRow = (t == numThreads - 1) ? n : startRow + chunkSize;
            threads.emplace_back(partiallyEliminateColumn, ref(a), i, startRow, endRow);
            startRow = endRow;
        }
        for (auto& t : threads) {
            t.join();
        }
    
        // Multiply the diagonal elements
        det *= a.data[(i * n) + i];
    }
    return det;
}