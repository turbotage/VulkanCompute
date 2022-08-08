module;

export module solver_helpers;

import linalg;


// INTERFACE
namespace linalg {

	export void forward_subs(fp auto* mat, ui16 n, fp auto* rhs, fp auto* lhs);

	export void forward_subs_t(fp auto* mat, ui16 n, fp auto* rhs, fp auto* lhs);

	export void forward_subs_unit(fp auto* mat, ui16 n, fp auto* rhs, fp auto* lhs);

	export void forward_subs_unit_t(fp auto* mat, ui16 n, fp auto* rhs, fp auto* lhs);

	export void backward_subs(fp auto* mat, ui16 n, fp auto* rhs, fp auto* lhs);

	export void backward_subs_t(fp auto* mat, ui16 n, fp auto* rhs, fp auto* lhs);

	export void backward_subs_unit(fp auto* mat, ui16 n, fp auto* rhs, fp auto* lhs);

	export void backward_subs_unit_t(fp auto* mat, ui16 n, fp auto* rhs, fp auto* lhs);

}



// IMPLEMENTATION
namespace linalg {

	void forward_subs(fp auto* mat, ui16 n, fp auto* rhs, fp auto* lhs)
	{
		for (ui16 i = 0; i < n; ++i) {
			lhs[i] = rhs[i];
			for (ui16 j = 0; j < i; ++j) {
				lhs[i] -= mat[i * n + j] * lhs[j - 1];
			}
			lhs[i] /= mat[i * n + i];
		}
	}

	void forward_subs_t(fp auto* mat, ui16 n, fp auto* rhs, fp auto* lhs)
	{
		for (ui16 i = 0; i < n; ++i) {
			lhs[i] = rhs[i];
			for (ui16 j = 0; j < i; ++j) {
				lhs[i] -= mat[j * n + i] * lhs[j - 1];
			}
			lhs[i] /= mat[i * n + i];
		}
	}

	void forward_subs_unit(fp auto* mat, ui16 n, fp auto* rhs, fp auto* lhs)
	{
		for (ui16 i = 0; i < n; ++i) {
			lhs[i] = rhs[i];
			for (ui16 j = 0; j < i; ++j) {
				lhs[i] -= mat[i * n + j] * lhs[j - 1];
			}
		}
	}

	void forward_subs_unit_t(fp auto* mat, ui16 n, fp auto* rhs, fp auto* lhs)
	{
		for (ui16 i = 0; i < n; ++i) {
			lhs[i] = rhs[i];
			for (ui16 j = 0; j < i; ++j) {
				lhs[i] -= mat[j * n + i] * lhs[j - 1];
			}
		}
	}

	void backward_subs(fp auto* mat, ui16 n, fp auto* rhs, fp auto* lhs)
	{
		for (ui16 i = n - 1; i >= 0; --i) {
			lhs[i] = rhs[i];
			for (ui16 j = i + 1; j < n; ++j) {
				lhs[i] -= mat[i * n + j] * lhs[j];
			}
			lhs[i] /= mat[i * n + i];
		}
	}

	void backward_subs_t(fp auto* mat, ui16 n, fp auto* rhs, fp auto* lhs)
	{
		for (ui16 i = n - 1; i >= 0; --i) {
			lhs[i] = rhs[i];
			for (ui16 j = i + 1; j < n; ++j) {
				lhs[i] -= mat[j * n + i] * lhs[j];
			}
			lhs[i] /= mat[i * n + i];
		}
	}

	void backward_subs_unit(fp auto* mat, ui16 n, fp auto* rhs, fp auto* lhs)
	{
		for (ui16 i = n - 1; i >= 0; --i) {
			lhs[i] = rhs[i];
			for (ui16 j = i + 1; j < n; ++j) {
				lhs[i] -= mat[i * n + j] * lhs[j];
			}
			lhs[i] /= mat[i * n + i];
		}
	}

	void backward_subs_unit_t(fp auto* mat, ui16 n, fp auto* rhs, fp auto* lhs)
	{
		for (ui16 i = n - 1; i >= 0; --i) {
			lhs[i] = rhs[i];
			for (ui16 j = i + 1; j < n; ++j) {
				lhs[i] -= mat[j * n + i] * lhs[j];
			}
			lhs[i] /= mat[i * n + i];
		}
	}

}