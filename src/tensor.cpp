#include "tensor.h"
#include "ops.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <numeric>

TensorData::TensorData(size_t size, Device device) : size(size), device(device) {
    if (device == Device::CPU) {
        data = new float[size]();
    } else {
        cudaError_t err = cudaMalloc(&data, size * sizeof(float));
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA malloc failed: " + std::string(cudaGetErrorString(err)));
        }
    }
}

TensorData::~TensorData() {
    if (device == Device::CPU) {
        delete[] data;
    } else {
        cudaFree(data);
    }
}

void TensorData::to_cpu() {
    if (device == Device::CPU) return;
    float* new_data = new float[size];
    cudaMemcpy(new_data, data, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(data);
    data = new_data;
    device = Device::CPU;
}

void TensorData::to_cuda() {
    if (device == Device::CUDA) return;
    float* new_data;
    cudaError_t err = cudaMalloc(&new_data, size * sizeof(float));
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA malloc failed");
    }
    cudaMemcpy(new_data, data, size * sizeof(float), cudaMemcpyHostToDevice);
    delete[] data;
    data = new_data;
    device = Device::CUDA;
}

Tensor::Tensor(const std::vector<int>& shape, Device device) : shape(shape) {
    size_t sz = 1;
    for (int s : shape) sz *= s;
    tensor_data = std::make_shared<TensorData>(sz, device);
}

Tensor::Tensor(const std::vector<float>& data, const std::vector<int>& shape, Device device) : shape(shape) {
    size_t sz = 1;
    for (int s : shape) sz *= s;
    if (data.size() != sz) {
        throw std::invalid_argument("Data size does not match shape");
    }
    tensor_data = std::make_shared<TensorData>(sz, Device::CPU);
    std::copy(data.begin(), data.end(), tensor_data->data);
    if (device == Device::CUDA) {
        tensor_data->to_cuda();
    }
}

size_t Tensor::size() const {
    return tensor_data->size;
}

Tensor Tensor::to(Device device) const {
    // Create new tensor with copied underlying data
    Tensor result(shape, device);
    if (device == get_device()) {
        if (device == Device::CPU) {
            std::copy(data_ptr(), data_ptr() + size(), result.data_ptr());
        } else {
            cudaMemcpy(result.data_ptr(), data_ptr(), size() * sizeof(float), cudaMemcpyDeviceToDevice);
        }
    } else if (device == Device::CPU) { // CUDA -> CPU
        cudaMemcpy(result.data_ptr(), data_ptr(), size() * sizeof(float), cudaMemcpyDeviceToHost);
    } else { // CPU -> CUDA
        cudaMemcpy(result.data_ptr(), data_ptr(), size() * sizeof(float), cudaMemcpyHostToDevice);
    }
    return result;
}

std::vector<float> Tensor::to_vector() const {
    std::vector<float> vec(size());
    if (get_device() == Device::CPU) {
        std::copy(data_ptr(), data_ptr() + size(), vec.begin());
    } else {
        cudaMemcpy(vec.data(), data_ptr(), size() * sizeof(float), cudaMemcpyDeviceToHost);
    }
    return vec;
}

Tensor Tensor::add(const Tensor& other) const {
    if (shape != other.shape) throw std::invalid_argument("Shape mismatch");
    if (get_device() != other.get_device()) throw std::invalid_argument("Device mismatch");
    Tensor result(shape, get_device());
    if (get_device() == Device::CPU) {
        for (size_t i = 0; i < size(); ++i) result.data_ptr()[i] = data_ptr()[i] + other.data_ptr()[i];
    } else {
        launch_add(data_ptr(), other.data_ptr(), result.data_ptr(), size());
    }
    return result;
}

Tensor Tensor::sub(const Tensor& other) const {
    if (shape != other.shape) throw std::invalid_argument("Shape mismatch");
    if (get_device() != other.get_device()) throw std::invalid_argument("Device mismatch");
    Tensor result(shape, get_device());
    if (get_device() == Device::CPU) {
        for (size_t i = 0; i < size(); ++i) result.data_ptr()[i] = data_ptr()[i] - other.data_ptr()[i];
    } else {
        launch_sub(data_ptr(), other.data_ptr(), result.data_ptr(), size());
    }
    return result;
}

Tensor Tensor::mul(const Tensor& other) const {
    if (shape != other.shape) throw std::invalid_argument("Shape mismatch");
    if (get_device() != other.get_device()) throw std::invalid_argument("Device mismatch");
    Tensor result(shape, get_device());
    if (get_device() == Device::CPU) {
        for (size_t i = 0; i < size(); ++i) result.data_ptr()[i] = data_ptr()[i] * other.data_ptr()[i];
    } else {
        launch_mul(data_ptr(), other.data_ptr(), result.data_ptr(), size());
    }
    return result;
}

Tensor Tensor::matmul(const Tensor& other) const {
    if (shape.size() != 2 || other.shape.size() != 2) throw std::invalid_argument("Matmul requires 2D tensors");
    int M = shape[0];
    int K = shape[1];
    int N = other.shape[1];
    if (other.shape[0] != K) throw std::invalid_argument("Inner dimensions must match");
    if (get_device() != other.get_device()) throw std::invalid_argument("Device mismatch");
    
    Tensor result({M, N}, get_device());
    if (get_device() == Device::CPU) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    sum += data_ptr()[i * K + k] * other.data_ptr()[k * N + j];
                }
                result.data_ptr()[i * N + j] = sum;
            }
        }
    } else {
        launch_matmul(data_ptr(), other.data_ptr(), result.data_ptr(), M, K, N);
    }
    return result;
}

void Tensor::fill(float value) {
    if (get_device() == Device::CPU) {
        for (size_t i = 0; i < size(); ++i) data_ptr()[i] = value;
    } else {
        launch_fill(data_ptr(), value, size());
    }
}
