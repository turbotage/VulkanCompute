module;

export module vc;

import <cmath>;
import <optional>;
import <vector>;
import <stdexcept>;
import <memory>;

export namespace vc {

	typedef ::std::uint64_t ui64;
	typedef ::std::uint32_t ui32;
	typedef ::std::uint16_t ui16;
	typedef ::std::uint8_t ui8;

	typedef ::std::int64_t i64;
	typedef ::std::int32_t i32;
	typedef ::std::int16_t i16;
	typedef ::std::int8_t i8;

	typedef ::std::uint_fast64_t ui64f;
	typedef ::std::uint_fast32_t ui32f;
	typedef ::std::uint_fast16_t ui16f;
	typedef ::std::uint_fast8_t ui8f;

	typedef ::std::int_fast64_t i64f;
	typedef ::std::int_fast32_t i32f;
	typedef ::std::int_fast16_t i16f;
	typedef ::std::int_fast8_t i8f;

	template<typename T>
	class raw_ptr {
	public:

		raw_ptr() { m_Ptr = nullptr; }
		raw_ptr(T& in) { m_Ptr = &in; }
		raw_ptr(const raw_ptr&) = delete;
		
		raw_ptr& operator=(const raw_ptr&) = delete;

		raw_ptr(raw_ptr&& other) {
			m_Ptr = other.m_Ptr;
			other.m_Ptr = nullptr;
		}
		void operator=(raw_ptr&& other) {
			m_Ptr = other.m_Ptr;
			other.m_Ptr = nullptr;
		}

		bool is_null() {
			return m_Ptr == nullptr;
		}

		T* get() { return m_Ptr; }

		T* operator->() { return m_Ptr; }

		T* operator->() const { return m_Ptr; }

		T& operator*() { return *m_Ptr; }

	private:
		T* m_Ptr;
	};

	// Used to signal output, functions with these parameters will fill the variable which the
	// reference points to
	template<typename T>
	using out_ref = T&;

	template<typename T>
	using refw = std::reference_wrapper<T>;

	// Used to signal output, functions with these parameters will fill the variable which the
	// reference points to if tc::OptOutRef isn't std::nullopt
	template<typename T>
	using opt_out_ref = std::optional<refw<T>>;

	template<typename T, typename U>
	using opt_out_pair_ref = std::optional<std::pair<refw<T>, refw<U>>>;

	template<typename T>
	using opt_ref = std::optional<refw<T>>;

	template<typename T, typename U>
	using opt_pair_ref = std::optional<std::pair<refw<T>, refw<U>>>;

	template<typename T>
	using opt_u_ptr = std::optional<std::unique_ptr<T>>;

	template<typename T>
	using opt_s_ptr = std::optional<std::shared_ptr<T>>;

	template <typename T>
	struct reversion_wrapper { T& iterable; };

	template <typename T>
	auto begin(reversion_wrapper<T> w) { return std::rbegin(w.iterable); }

	template <typename T>
	auto end(reversion_wrapper<T> w) { return std::rend(w.iterable); }

	template <typename T>
	reversion_wrapper<T> reverse(T&& iterable) { return { iterable }; }

	std::vector<int64_t> tc_broadcast_shapes(const std::vector<int64_t>& shape1, const std::vector<int64_t>& shape2)
	{
		if (shape1.size() == 0 || shape2.size() == 0)
			throw std::runtime_error("shapes must have atleast one dimension to be broadcastable");

		auto& small_shape = (shape1.size() > shape2.size()) ? shape2 : shape1;
		auto& big_shape = (shape1.size() > shape2.size()) ? shape1 : shape2;

		std::vector<int64_t> ret(big_shape.size());

		auto retit = ret.rbegin();
		auto smallit = small_shape.rbegin();
		for (auto bigit = big_shape.rbegin(); bigit != big_shape.rend(); ) {
			if (smallit != small_shape.rend()) {
				if (*smallit == *bigit) {
					*retit = *bigit;
				}
				else if (*smallit > *bigit && *bigit == 1) {
					*retit = *smallit;
				}
				else if (*bigit > *smallit && *smallit == 1) {
					*retit = *bigit;
				}
				else {
					throw std::runtime_error("shapes where not broadcastable");
				}
				++smallit;
			}
			else {
				*retit = *bigit;
			}

			++bigit;
			++retit;
		}

		return ret;
	}

}