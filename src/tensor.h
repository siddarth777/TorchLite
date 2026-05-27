#pragma once
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <numeric>

enum class Device { CPU, CUDA };

class TensorData {
public:
    float* data;
    size_t size;
    Device device;

    TensorData(size_t size, Device device);
    ~TensorData();

    // Disable copy
    TensorData(const TensorData&) = delete;
    TensorData& operator=(const TensorData&) = delete;

    void to_cpu();
    void to_cuda();
};

class Tensor {
public:
    std::vector<int> shape;
    std::shared_ptr<TensorData> tensor_data;
    
    Tensor(const std::vector<int>& shape, Device device = Device::CPU);
    Tensor(const std::vector<float>& data, const std::vector<int>& shape, Device device = Device::CPU);

    size_t size() const;
    const std::vector<int>& get_shape() const { return shape; }
    Device get_device() const { return tensor_data->device; }
    
    Tensor to(Device device) const;
    std::vector<float> to_vector() const;
    
    float* data_ptr() const { return tensor_data->data; }

    // Ops
    Tensor add(const Tensor& other) const;
    Tensor sub(const Tensor& other) const;
    Tensor mul(const Tensor& other) const;
    Tensor matmul(const Tensor& other) const;
    Tensor transpose() const;
    
    void fill(float value);
};
