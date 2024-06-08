#include <iostream>
#include <thread>
#include <immintrin.h>
#include <chrono>
#include <iomanip>
#include <immintrin.h> // SIMD
#include <cstring>
#include <mutex>
#include <condition_variable>

class Matrix {
    private:
        int** M;
        int n, m;

    public:
        Matrix(int a, int b);
        ~Matrix();
        void initialiseMatrix();
        void inputMatrix();
        void displayMatrix();
        int** T();
        Matrix* multiplyMatrix(Matrix* N);

        // helper function
        void set (int i, int j, int num) {
            M[i][j] = num;
        }
        
        // method 1
        static void sum_up(const Matrix &A, const Matrix &B, Matrix &C, const int start_row, const int num_rows, const int dim){
            int variable_that_is_not_shared;
            for(int i = start_row; i < (start_row + num_rows); i++){
                for(int j = 0; j < B.m; j++){
                    variable_that_is_not_shared = 0;
                    for(int k = 0; k < dim; k++){
                        variable_that_is_not_shared += A.M[i][k] * B.M[k][j];
                    }
                    C.M[i][j] = variable_that_is_not_shared;
                }
            }
        }

        // method 2
        /*
        static void sum_up(const Matrix &A, const Matrix &B, Matrix &C, const int elements, const int start_index, const int dim){
            int i, j, s, variable_that_is_not_shared;
            s = B.m;
            for(int l = start_index; l < start_index + elements; l++){
                variable_that_is_not_shared = 0;
                for(int k = 0; k < dim; k++){
                    i = l / s;
                    j = l % s;
                    variable_that_is_not_shared += A.M[i][k] * B.M[k][j];
                }
                C.M[i][j] = variable_that_is_not_shared;
            }
        }
        */
};