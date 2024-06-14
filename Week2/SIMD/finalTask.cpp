#include <iostream>
#include <immintrin.h>
#include <iomanip>
#include <chrono>
using namespace std;
/*

COMPUTE DOT PRODUCT OF TWO VECTORS
INPUTS :n => dimension of vectors
        vec1 => the first vector
        vec2 => the second vector

OUTPUT:dotProduct = vec1.vec2

dotProduct of two vectors is defined as summation_{i=0}^{i=n} [vec1[i]*vec2[i]];

*/
int main () {

    int n;
    cin>>n;
    float vec1[n];
    float vec2[n];
    for (int i=0; i<n; i++) {
        cin>>vec1[i];
    }
    for (int i=0; i<n; i++) {
        cin>>vec2[i];
    }
    float naiveTimeTaken = 0;
    float SIMDTimeTaken = 0;
    float dotProduct = 0;
    for (int i=0; i<1000; i++) {
        dotProduct = 0;
        auto start = chrono::high_resolution_clock::now();
        //NAIVE IMPLEMENTATION FOR COMPARISION PURPOSES
        for (int i=0; i<n; i++) {
            dotProduct += vec1[i]*vec2[i];
        }
        //NAIVE IMPLEMENTATION ENDS
        auto end = chrono::high_resolution_clock::now();
        auto elapsed = chrono::duration_cast<chrono::duration<double>>(end - start);
        naiveTimeTaken += (elapsed.count())/1000;
    }

    std::cout<<dotProduct<<" "<<naiveTimeTaken<<" ";
    
    for (int i=0; i<1000; i++) {
        dotProduct = 0;
        auto start2 = chrono::high_resolution_clock::now();
        //STUDENT CODE BEGINS HERE
        __m256 v1, v2, v3;
        __m128 v4, v5, v6;
        float vec3[4];
        for (int i = 0; i < ((n % 8) ? (n - 8) : n); i += 8) {
            v1 = _mm256_loadu_ps(vec1 + i);
            v2 = _mm256_loadu_ps(vec2 + i);
            v3 = _mm256_mul_ps(v1, v2);
            v4 = _mm256_extractf128_ps(v3, 0);
            v5 = _mm256_extractf128_ps(v3, 1);
            v6 = _mm_hadd_ps(v4, v5);
            _mm_storeu_ps(vec3, v6);
            for (int i = 0; i < 4; i++) {
                dotProduct += vec3[i];
            }
        }
        for (int i = n - (n % 8); i < n; i++) {
            dotProduct += vec1[i] * vec2[i];
        }
        //END OF STUDENT CODE
        auto end2 = chrono::high_resolution_clock::now();
        auto elapsed2 = chrono::duration_cast<chrono::duration<double>>(end2 - start2);
        SIMDTimeTaken += (elapsed2.count())/1000;
    }
    std::cout<<dotProduct<<" "<<SIMDTimeTaken<<" ";
}