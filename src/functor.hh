#pragma once
#include <iostream>
struct Point2
{
    Point2() : x(0), y(0) {}
    Point2(float a, float b) : x(a), y(b) {}
    float x;
    float y;
};

std::ostream& operator<<(std::ostream& os, Point2 pt) {
    os << pt.x << "," << pt.y;
    return os;
}

struct dist2 {
    __host__ __device__ float operator()(Point2 input) {
        return sqrt(input.x * input.x + input.y * input.y);
    }
};

struct float_plus {
    __host__ __device__ float operator()(float a, float b) {
        return a+b;
    }
};