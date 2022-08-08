module;

#include <type_traits>
#include <concepts>
#include <memory>
#include <string>
#include <optional>

export module linalg;

import glsl;
import util;

export using ui8 = std::uint8_t;
export using ui16 = std::uint16_t;
export using ui32 = std::uint32_t;
export using ui64 = std::uint64_t;


export template<class T>
concept fp = std::is_floating_point<T>::value;

// INTERFACE
namespace linalg {
	
	inline void swap(fp auto& elm1, fp auto& elm2);

	inline fp auto abs(fp auto& val);

	export void square_transpose_i(fp auto* mat, ui16 n);

	export void transpose(fp auto* inmat, fp auto* outmat, ui16 n);

	export std::string glsl_transpose(int nrows, int mcols, bool single_precission = true);

	export void transpose_submatrix(fp auto* mat, ui16 n, ui16 subidx = 0);

	export void row_interchange(fp auto* mat, ui16 n, ui16 ii, ui16 jj);

	export void column_interchange(fp auto* mat, ui16 n, ui16 ii, ui16 jj);

	export void row_column_interchange(fp auto* mat, ui16 n, ui16 ii, ui16 jj);

	export void column_row_interchange(fp auto* mat, ui16 n, ui16 ii, ui16 jj);

	export void get_max_diagonal(fp auto* mat, ui16 n, ui16& max_idx, fp auto& max_val, ui16 submat_idx = 0);

	export void get_mag_diagonal(fp auto* mat, ui16 n, ui16& max_idx, fp auto& max_val, ui16 submat_idx = 0);

	export void pivot_max_diagonal(fp auto* mat, ui16 n, fp auto* perm);

	export void pivot_mag_diagonal(fp auto* mat, ui16 n, fp auto* perm);

	export void mul_diag_vec(fp auto* mat, ui16 n, fp auto* vec);

	export void mul_inv_diag_vec(fp auto* mat, ui16 n, fp auto* vec);

	export void mul_mat_mat(fp auto* rmat, ui16 rn, ui16 rm, 
							fp auto* lmat, ui16 ln, ui16 lm, 
							fp auto* omat);

	export void mul_mat_vec(fp auto* mat, ui16 n, ui16 m, fp auto* vec, fp auto* ovec);


}


// IMPLEMENTATION
namespace linalg {

	inline void swap(fp auto& elm1, fp auto& elm2)
	{
		auto temp = elm1;
		elm1 = elm2;
		elm2 = temp;
	}

	inline fp auto abs(const fp auto& val)
	{
		return val > 0.0 ? val : -val;
	}

	void square_transpose_i(fp auto* mat, ui16 n) 
	{
		using T = std::remove_reference<decltype(mat)>::type;
		T temp;
		for (ui16 i = 0; i < n; ++i) {
			for (ui16 j = 0; j < n; ++j) {
				swap(&mat[i * n + j], &mat[j * n + i]);
			}
		}
	}

	void transpose_submatrix(fp auto* mat, ui16 n, ui16 subidx)
	{
		using T = std::remove_reference<decltype(mat)>::type;
		T temp;
		for (ui16 i = subidx; i < n; ++i) {
			for (ui16 j = subidx; j < n; ++j) {
				swap(&mat[i * n + j], &mat[j * n + i]);
			}
		}
	}

	void row_interchange(fp auto* mat, ui16 n, ui16 ii, ui16 jj)
	{
		for (ui16 k = 0; k < n; ++k) {
			swap(&mat[ii * n + k], &mat[jj * n + k]);
		}
	}

	void column_interchange(fp auto* mat, ui16 n, ui16 ii, ui16 jj)
	{
		for (ui16 k = 0; k < n; ++k) {
			swap(&mat[k * n + ii], &mat[k * n + jj]);
		}
	}

	void row_column_interchange(fp auto* mat, ui16 n, ui16 ii, ui16 jj)
	{
		row_interchange(mat, n, ii, jj);
		column_interchange(mat, n, ii, jj);
	}

	void column_row_interchange(fp auto* mat, ui16 n, ui16 ii, ui16 jj)
	{
		column_interchange(mat, n, ii, jj);
		row_interchange(mat, n, ii, jj);
	}

	void get_max_diagonal(fp auto* mat, ui16 n, ui16& max_idx, fp auto& max_val, ui16 submat_idx)
	{
		max_val = 0.0;
		max_idx = 0;

		for (int i = submat_idx; i < n; ++i) {
			auto val = mat[i * n + i];
			if (val > max_val) {
				max_val = val;
				max_idx = i;
			}
		}
	}

	void get_mag_diagonal(fp auto* mat, ui16 n, ui16& max_idx, fp auto& max_val, ui16 submat_idx)
	{
		max_val = 0.0;
		max_idx = 0;

		for (int i = submat_idx; i < n; ++i) {
			auto val = abs(mat[i * n + i]);
			if (val > max_val) {
				max_val = val;
				max_idx = i;
			}
		}
	}

	void pivot_max_diagonal(fp auto* mat, ui16 n, fp auto* perm)
	{
		using T = std::remove_reference<decltype(mat)>::type;

		T max_val;
		ui16 max_idx;

		for (ui16 i = 0; i < n; ++i) {
			get_max_diagonal(mat, n, max_idx, max_val, i);

			if (max_idx != i) {
				row_column_interchange(mat, n, i, max_idx);

				swap(perm[i], perm[max_idx]);
			}
		}
	}

	void pivot_mag_diagonal(fp auto* mat, ui16 n, fp auto* perm)
	{
		using T = std::remove_reference<decltype(mat)>::type;

		T max_val;
		ui16 max_idx;

		for (ui16 i = 0; i < n; ++i) {
			get_mag_diagonal(mat, n, max_idx, max_val, i);

			if (max_idx != i) {
				row_column_interchange(mat, n, i, max_idx);

				swap(perm[i], perm[max_idx]);
			}
		}
	}

	void mul_diag_vec(fp auto* mat, ui16 n, fp auto* vec)
	{
		for (ui16 i = 0; i < n; ++i) {
			vec[i] *= mat[i * n + i];
		}
	}

	void mul_inv_diag_vec(fp auto* mat, ui16 n, fp auto* vec)
	{
		for (ui16 i = 0; i < n; ++i) {
			vec[i] /= mat[i * n + i];
		}
	}

	void mul_mat_mat(fp auto* rmat, ui16 rn, ui16 rm,
						fp auto* lmat, ui16 ln, ui16 lm,
						fp auto* omat)
	{
		using T = std::remove_reference<decltype(rmat)>::type;

		for (ui16 i = 0; i < rn; ++i) {
			for (ui16 j = 0; j < lm; ++j) {
				T entry = 0.0;

				for (ui16 k = 0; k < rm; ++k) {
					entry += lmat[i * lm + k] * rmat[k * rm + j];
				}

				omat[i * lm + j] = entry;
			}
		}
	}

	void mul_mat_vec(fp auto* omat, ui16 n, ui16 m, fp auto* ovec)
	{
		using T = std::remove_reference<decltype(omat)>::type;

		for (ui16 i = 0; i < n; ++i) {
			T entry = 0.0;

			for (ui16 j = 0; j < m; ++j) {
				entry += omat[i * m + j] * ovec[j];
			}

			ovec[i] = entry;
		}
	}

}
