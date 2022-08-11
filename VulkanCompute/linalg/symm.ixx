module;

#include <vector>

export module symm;

import linalg;

// INTERFACE
/*
namespace linalg {
namespace symm {
	
	// LDL

	export void ldl(fp auto* mat, ui16 n);

	export void solve_ldl(fp auto* mat, ui16 n, fp auto* rhs);

	export void solve_perm_ldl(fp auto* mat, ui16 n, fp auto* rhs, fp auto* perm);

	// GMW81

	export void gmw81(fp auto* mat, ui16 n);

	export void solve_gmw81(fp auto* mat, ui16 n, fp auto* rhs);

	export void solve_perm_gmw81(fp auto* mat, ui16 n, fp auto* rhs, fp auto* perm);

}
}


// IMPLEMENTATION
namespace linalg {
namespace symm {

	void ldl(fp auto* mat, ui16 n)
	{
		using T = std::remove_pointer<decltype(mat)>::type;

		static std::vector<T> arr;

		if (arr.size() != n)
			arr.resize(n);

		for (ui16 i = 0; i < n; ++i) {
			
			auto d = mat[i * n + i];

			for (ui16 j = i + 1; j < n; ++j) {
				auto& matl = mat[j * n + i];
				arr[j] = matl;
				matl /= d;
			}

			for (ui16 j = i + 1; j < n; ++j) {
				auto aj = arr[j];
				for (ui16 k = j; k < n; ++k) {
					mat[k * n + j] -= aj * mat[k * n + i];
				}
			}

		}
	}

	void gmw81(fp auto* mat, ui16 n)
	{
		using T = std::remove_pointer<decltype(mat)>::type;

		T m1 = 0.0;
		T m2 = 0.0;
		T beta2 = 0.0;

		T temp;

		static std::vector<T> arr;

		if (arr.size() != n)
			arr.resize(n);

		for (ui16 i = 0; i < n; ++i) {
			temp = mat[i * n + i];
			temp = temp > 0.0 ? temp : -temp;

			if (m1 < temp)
				m1 = temp;
		}

		if (beta2 < m1)
			beta2 = m1;

		for (ui16 i = 1; i < n; ++i) {
			for (ui16 j = 0; j < i; ++j) {
				temp = mat[i * n + j];
				temp = temp > 0.0 ? temp : -temp;

				if (m2 < temp)
					m2 = temp;
			}
		}

		if (n > 1)
			m2 /= (T)sqrt(n * n - 1);

		if (beta2 < m2)
			beta2 = m2;

		for (ui16 i = 0; i < n; ++i) {
			temp = mat[i * n + i];
			decltype((*mat)) d = temp > 0.0 ? temp : -temp;

			if (d < std::numeric_limits<T>::epsilon())
				d = std::numeric_limits<T>::epsilon();

			m2 = 0.0;
			for (ui16 j = i + 1; j < n; ++j) {
				temp = mat[j * n + i];
				temp = temp > 0.0 ? temp : -temp;

				if (m2 < temp)
					m2 = temp;
			}

			m2 *= m2;

			if (m2 > d * beta2)
				d = m2 / beta2;

			mat[i * n + i] = d;

			for (ui16 j = i + 1; j < n; ++j) {
				arr[j] = mat[j * n + i];
				mat[j * n + i] /= d;

				for (ui16 k = j; k < n; ++k) {
					mat[k, j] -= arr[j] * mat[k, i];
				}
			}

		}

	}

}
}


*/