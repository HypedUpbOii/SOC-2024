#include "matrix.h"
#include <vector>
#define Loop(i,a,b) for (int i = a ; i < b ; i++)
#define MAX_THREADS 20
using namespace std;

Matrix::Matrix(int a, int b) { // generate a matrix (2D array) of dimensions a,b
    this->n = a;
    this->m = b;
    this->M = new int*[a];
    Loop(i, 0, n) this->M[i] = new int[b];
    this->initialiseMatrix();
}

Matrix::~Matrix() { // cleanup heap memory
    Loop(i, 0, this->n) delete[] this->M[i];
    delete[] this->M;
}

void Matrix::initialiseMatrix(){ // initialise entries to 0
    Loop(i, 0, this->n) {
        Loop(j, 0, this->m) this->M[i][j] = 0;
    }
}

void Matrix::inputMatrix() { // take input
    Loop(i, 0, this->n) {
        Loop(j, 0, this->m) cin >> this->M[i][j];
    }
}

void Matrix::displayMatrix() { // print matrix
    Loop(i, 0, this->n) {
        Loop(j, 0, this->m) cout << this->M[i][j] << " ";
        cout << "\n";
    }
}
int** Matrix::T(){
    int** MT = new int*[this->m];
    Loop(i,0,m) MT[i] = new int[this->n];
    Loop(i,0,m){
        Loop(j,0,n){
            MT[i][j] = this->M[j][i];
        }
    }
    return MT;
}

Matrix* Matrix::multiplyMatrix(Matrix* N) {
    if (this->m != N->n) {
        return NULL;
    }
    Matrix *c = new Matrix(this->n,N->m);
    vector<thread> threads;
    
    // method 1
    int num_threads = min(MAX_THREADS, this->n);
    int start_row = 0;
    int remaining = this->n % num_threads;
    int num_rows = this->n / num_threads;
    if (remaining) num_rows++;
    for(int i = 0; i < num_threads; i++){
        if ((remaining > 0) && (i > remaining - 1)) num_rows--;
        threads.push_back(thread(&Matrix::sum_up, ref(*this), ref(*N), ref(*c), start_row, num_rows, this->m));
        start_row += num_rows;
    }

    // method 2
    /*
    int x, y, a, b;
    y = (this->n * N->m) - (MAX_THREADS * ((this->n * N->m) / MAX_THREADS));
    x = MAX_THREADS - y;
    a = (this->n * N->m) / MAX_THREADS;
    b = this->m;
    for(int i = 0; i < y; i++){
        threads.push_back(thread(&Matrix::sum_up, ref(*this), ref(*N), ref(*c), a + 1, i * (a + 1), b));
    }
    for(int i = 0; i < x; i++){
        threads.push_back(thread(&Matrix::sum_up, ref(*this), ref(*N), ref(*c), a, (y * (a + 1)) + (i * a), b));
    }
    */

    for(auto& t : threads){
        if(t.joinable()){
            t.join();
        }   
    }
    return c;
}