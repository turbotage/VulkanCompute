
void copy_mat(out FP_TYPE out_mat[NDIM*NDIM], in  FP_TYPE in_mat[NDIM]) {
	for (int i = 0; i < NDIM; ++i) {
		for (int j = 0; j < NDIM; ++j) {
			out_mat[i*NDIM + j] = in_mat[i*NDIM + j];
		}
	}
}
