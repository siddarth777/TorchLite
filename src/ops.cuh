#pragma once
#include <stddef.h>

void launch_add(const float* a, const float* b, float* out, size_t size);
void launch_sub(const float* a, const float* b, float* out, size_t size);
void launch_mul(const float* a, const float* b, float* out, size_t size);
void launch_matmul(const float* a, const float* b, float* out, int M, int K, int N);
void launch_fill(float* data, float value, size_t size);
