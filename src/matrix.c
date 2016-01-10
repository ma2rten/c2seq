#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <cblas.h>

void print_matrix(float* A, int dim1, int dim2) {
    for(int i = 0; i < dim1; i++) {
        for(int j = 0; j < dim2; j++) {
            printf("%f ", A[i * dim2 + j]);
        }
        printf("\n");
    }
}

void print_vector(float* A, int dim1) {
    printf("[");
    for(int i = 0; i < dim1; i++) {
        if(i > 0)
            printf(", ");
        printf("%.8f", A[i]);
    }
    printf("]\n");
}

int argmax(float* X, int len)
{
    float max_val = X[0];
    int idx = 0;

    for(int i = 1; i < len; i++) {
        if(max_val < X[i]) {
            idx = i;
            max_val = X[i];
        }
    }

    return idx;
}

// A += np.outer(x,y)
void outer_add(float* A, int A_dim1, int A_dim2, float* x, int x_dim, float* y, int y_dim) {
    assert(A_dim1 == x_dim && A_dim2 == y_dim);
    cblas_sger(CblasColMajor, x_dim, y_dim, 1.0f, x, 1, y, 1, A, x_dim);
}

// A = np.outer(x,y)
void outer(float* A, int A_dim1, int A_dim2, float* x, int x_dim, float* y, int y_dim) {
    memset(A, 0, sizeof(float) * A_dim1 * A_dim2);
    outer_add(A, A_dim1, A_dim2, x, x_dim, y, y_dim);
}

// y = np.dot(A, x)
void dot(float* y, int y_dim, float* A, int A_dim1, int A_dim2, float* x, int x_dim) {
    assert(A_dim1 == y_dim && A_dim2 == x_dim);
    cblas_sgemv(CblasColMajor, CblasNoTrans, A_dim1, A_dim2, 1.0f, A, y_dim, x, 1, 0.0f, y, 1);
}

// y += np.dot(A, x)
void dot_add(float* y, int y_dim, float* A, int A_dim1, int A_dim2, float* x, int x_dim) {
    assert(A_dim1 == y_dim && A_dim2 == x_dim);
    cblas_sgemv(CblasColMajor, CblasNoTrans, A_dim1, A_dim2, 1.0f, A, y_dim, x, 1, 1.0f, y, 1);
}

// y = np.dot(A.T, x)
void dot_trans(float* y, int y_dim, float* A, int A_dim1, int A_dim2, float* x, int x_dim) {
    assert(A_dim1 == x_dim && A_dim2 == y_dim);
    cblas_sgemv(CblasColMajor, CblasTrans, A_dim1, A_dim2, 1.0f, A, A_dim1, x, 1, 0.0f, y, 1);
}

// y += np.dot(A.T, x)
void dot_trans_add(float* y, int y_dim, float* A, int A_dim1, int A_dim2, float* x, int x_dim) {
    assert(A_dim1 == x_dim && A_dim2 == y_dim);
    cblas_sgemv(CblasColMajor, CblasTrans, A_dim1, A_dim2, 1.0f, A, A_dim1, x, 1, 1.0f, y, 1);
}
