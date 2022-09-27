
void gmw81(inout FP_TYPE mat[NDIM*NDIM], inout FP_TYPE work_arr[NDIM]) {

	FP_TYPE m1 = 0.0;
	FP_TYPE m2 = 0.0;
	FP_TYPE beta2 = 0.0;

	FP_TYPE d;
	FP_TYPE temp;

	for (int i = 0; i < NDIM; ++i) {
		temp = mat[i*NDIM + i];
		temp = temp > 0.0 ? temp : -temp;
		if (m1 < temp) {
			m1 = temp;
		}
	}

	if (beta2 < m1)
		beta2 = m1;

	for (int i = 1; i < NDIM; ++i) {
		for (int j = 0; j < i; ++j) {
			temp = abs(A[i*NDIM + j]);
			if (m2 < temp) {
				m2 = temp;
			}
		}
	}

	if (n > 1) {
		m2 /= sqrt(NDIM*NDIM -1)
	}

	if (beta2 < m2) {
		beta2 = m2;
	}

	for (int i = 0; i < NDIM; ++i) {
		d = abs(A[i*NDIM + i]);

		if (d < DELTA) {
			d = DELTA;
		}

		m2 = 0.0;
		for (int j = i + 1; j < n; ++j) {
			temp = abs(A[j*NDIM + i]);
			if (m2 < temp) {
				m2 = temp;
			}
		}

		m2 *= m2;

		if (m2 > d*beta2) {
			d = m2 / beta2;
		}
		
		A[i*NDIM + i] = d;

		for (int j = i+1; j < NDIM; ++j) {
			work_arr[j] = A[j*NDIM + i];
			A[j*NDIM + i] /= d;
			for (int k = j; k < NDIM; ++k) {
				A[k*NDIM + j] -= work_arr[k] * A[k*NDIM + i];
			}
		}

	}

}
