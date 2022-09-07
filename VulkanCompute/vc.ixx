module;

#include <cmath>
#include <optional>
#include <vector>
#include <stdexcept>
#include <memory>

export module vc;

export namespace vc {

	typedef ::std::uint_fast64_t ui64;
	typedef ::std::uint_fast32_t ui32;
	typedef ::std::uint_fast16_t ui16;
	typedef ::std::uint_fast8_t ui8;

	typedef ::std::int_fast64_t i64;
	typedef ::std::int_fast32_t i32;
	typedef ::std::int_fast16_t i16;
	typedef ::std::int_fast8_t i8;

	// Used to signal output, functions with these parameters will fill the variable which the
	// reference points to
	template<typename T>
	using OutRef = T&;

	template<typename T>
	using refw = std::reference_wrapper<T>;

	// Used to signal output, functions with these parameters will fill the variable which the
	// reference points to if tc::OptOutRef isn't std::nullopt
	template<typename T>
	using OptOutRef = std::optional<refw<T>>;

	template<typename T, typename U>
	using OptOutPairRef = std::optional<std::pair<refw<T>, refw<U>>>;

	template<typename T>
	using OptRef = std::optional<refw<T>>;

	template<typename T, typename U>
	using OptPairRef = std::optional<std::pair<refw<T>, refw<U>>>;

	template<typename T>
	using OptUPtr = std::optional<std::unique_ptr<T>>;

	template<typename T>
	using OptSPtr = std::optional<std::shared_ptr<T>>;

	enum eBuildMode {
		Release,
		Debug
	};

	constexpr ui8 BUILD_MODE = eBuildMode::Debug;
	/*
	Statements such as
	if constexpr (BUILD_MODE == eBuildMode::Debug) {
		doSomething();
	}
	should be remove by Dead-Code-Compilation
	*/

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

		auto& small = (shape1.size() > shape2.size()) ? shape2 : shape1;
		auto& big = (shape1.size() > shape2.size()) ? shape1 : shape2;

		std::vector<int64_t> ret(big.size());

		auto retit = ret.rbegin();
		auto smallit = small.rbegin();
		for (auto bigit = big.rbegin(); bigit != big.rend(); ) {
			if (smallit != small.rend()) {
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