
void print_matrix(float* A, int dim1, int dim2);
void print_vector(float* A, int dim1);


int argmax(float* X, int len);

// A = np.outer(x,y)
void outer(float* A, int A_dim1, int A_dim2, float* x, int x_dim, float* y, int y_dim);

// A += np.outer(x,y)
void outer_add(float* A, int A_dim1, int A_dim2, float* x, int x_dim, float* y, int y_dim);

// y = np.dot(A, x)
void dot(float* y, int y_dim, float* A, int A_dim1, int A_dim2, float* x, int x_dim);

// y += np.dot(A, x)
void dot_add(float* y, int y_dim, float* A, int A_dim1, int A_dim2, float* x, int x_dim);

// y = np.dot(A.T, x)
void dot_trans(float* y, int y_dim, float* A, int A_dim1, int A_dim2, float* x, int x_dim);

// y += np.dot(A.T, x)
void dot_trans_add(float* y, int y_dim, float* A, int A_dim1, int A_dim2, float* x, int x_dim);
