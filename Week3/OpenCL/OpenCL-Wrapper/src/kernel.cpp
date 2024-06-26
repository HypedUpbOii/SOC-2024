#include "kernel.hpp" // note: unbalanced round brackets () are not allowed and string literals can't be arbitrarily long, so periodically interrupt with )+R(
string opencl_c_container() { return R( // ########################## begin of OpenCL C code ####################################################################



kernel void add_kernel(global float* A, global float* B, global float* C) { // equivalent to "for(uint n=0u; n<N; n++) {", but executed in parallel
	const uint n = get_global_id(0);
	C[n] = A[n]+B[n];
}

kernel void mul_kernel(global float* A, global float* B, global float* C) {
	// TASK 1 CODE BEGINS HERE
	const uint n = get_global_id(0);
	C[n] = A[n] * B[n];
	// TASK 1 CODE ENDS HERE
}

kernel void matMul (__global float* A, __global float *B, __global float *C, int aCol, int cRow, int cCol) {
	// TASK 2 CODE BEGINS HERE
	// HINT : IMPLEMENT DOT PRODUCT HERE
	const uint u = get_global_id(0); //prolly of size cRow * cCol
	//size of matrix A = cRow * aCol; size of matrix B = aCol * cCol
	int i = u / aCol; //C[i][j]
	int j = u % aCol;
	float dotPdt = 0.0f;
	for (uint k = 0; k < aCol; k++) {
		dotPdt += A[(i * aCol) + k] * B[(k * cRow) + j];
	}
	C[u] = dotPdt;
	// TASK 2 CODE ENDS HERE
}


);} // ############################################################### end of OpenCL C code #####################################################################
