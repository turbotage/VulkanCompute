module;

export module sub;

import <memory>;
import <stdexcept>;
import <complex>;

import token;
import vc;
import token_algebra;

namespace expression {

	// Sub

	std::unique_ptr<NumberBaseToken> operator-(const Token& a, const Token& b);

	std::unique_ptr<NumberBaseToken> operator-(const ZeroToken& a, const Token& b);
	std::unique_ptr<NumberBaseToken> operator-(const UnityToken& a, const Token& b);
	std::unique_ptr<NumberBaseToken> operator-(const NegUnityToken& a, const Token& b);
	std::unique_ptr<NumberBaseToken> operator-(const NanToken& a, const Token& b);
	std::unique_ptr<NumberBaseToken> operator-(const NumberToken& a, const Token& b);

	ZeroToken operator-(const ZeroToken& a, const ZeroToken& b);
	UnityToken operator-(const ZeroToken& a, const NegUnityToken& b);
	NegUnityToken operator-(const ZeroToken& a, const UnityToken& b);
	NanToken operator-(const ZeroToken& a, const NanToken& b);
	NumberToken operator-(const ZeroToken& a, const NumberToken& b);

	NegUnityToken operator-(const NegUnityToken& a, const ZeroToken& b);
	ZeroToken operator-(const NegUnityToken& a, const NegUnityToken& b);
	NumberToken operator-(const NegUnityToken& a, const UnityToken& b);
	NanToken operator-(const NegUnityToken& a, const NanToken& b);
	NumberToken operator-(const NegUnityToken& a, const NumberToken& b);

	UnityToken operator-(const UnityToken& a, const ZeroToken& b);
	NumberToken operator-(const UnityToken& a, const NegUnityToken& b);
	ZeroToken operator-(const UnityToken& a, const UnityToken& b);
	NanToken operator-(const UnityToken& a, const NanToken& b);
	NumberToken operator-(const UnityToken& a, const NumberToken& b);

	NanToken operator-(const NanToken& a, const ZeroToken& b);
	NanToken operator-(const NanToken& a, const NegUnityToken& b);
	NanToken operator-(const NanToken& a, const UnityToken& b);
	NanToken operator-(const NanToken& a, const NanToken& b);
	NanToken operator-(const NanToken& a, const NumberToken& b);

	NumberToken operator-(const NumberToken& a, const ZeroToken& b);
	NumberToken operator-(const NumberToken& a, const NegUnityToken& b);
	NumberToken operator-(const NumberToken& a, const UnityToken& b);
	NanToken operator-(const NumberToken& a, const NanToken& b);
	NumberToken operator-(const NumberToken& a, const NumberToken& b);




	// IMPLEMENTATION


	std::unique_ptr<NumberBaseToken> operator-(const Token& a, const Token& b)
	{
		switch (a.get_token_type()) {
		case TokenType::ZERO_TYPE:
		{
			const ZeroToken& atok = static_cast<const ZeroToken&>(a);
			return atok - b;
		}
		case TokenType::UNITY_TYPE:
		{
			const UnityToken& atok = static_cast<const UnityToken&>(a);
			return atok - b;
		}
		case TokenType::NEG_UNITY_TYPE:
		{
			const NegUnityToken& atok = static_cast<const NegUnityToken&>(a);
			return atok - b;
		}
		case TokenType::NAN_TYPE:
		{
			const NanToken& atok = static_cast<const NanToken&>(a);
			return atok - b;
		}
		case TokenType::NUMBER_TYPE:
		{
			const NumberToken& atok = static_cast<const NumberToken&>(a);
			return atok - b;
		}
		default:
			throw std::runtime_error("Negation can only be applied to tokens Zero, Unity, NegUnity, Nan and Number");
		}
	}

	std::unique_ptr<NumberBaseToken> operator-(const ZeroToken& a, const Token& b)
	{
		switch (b.get_token_type()) {
		case TokenType::ZERO_TYPE:
		{
			const ZeroToken& btok = static_cast<const ZeroToken&>(b);
			return to_ptr(a - btok);
		}
		case TokenType::UNITY_TYPE:
		{
			const UnityToken& btok = static_cast<const UnityToken&>(b);
			return to_ptr(a - btok);
		}
		case TokenType::NEG_UNITY_TYPE:
		{
			const NegUnityToken& btok = static_cast<const NegUnityToken&>(b);
			return to_ptr(a - btok);
		}
		case TokenType::NAN_TYPE:
		{
			const NanToken& btok = static_cast<const NanToken&>(b);
			return to_ptr(a - btok);
		}
		case TokenType::NUMBER_TYPE:
		{
			const NumberToken& btok = static_cast<const NumberToken&>(b);
			return to_ptr(a - btok);
		}
		default:
			throw std::runtime_error("Negation can only be applied to tokens Zero, Unity, NegUnity, Nan and Number");
		}
	}

	std::unique_ptr<NumberBaseToken> operator-(const UnityToken& a, const Token& b)
	{
		switch (b.get_token_type()) {
		case TokenType::ZERO_TYPE:
		{
			const ZeroToken& btok = static_cast<const ZeroToken&>(b);
			return to_ptr(a - btok);
		}
		case TokenType::UNITY_TYPE:
		{
			const UnityToken& btok = static_cast<const UnityToken&>(b);
			return to_ptr(a - btok);
		}
		case TokenType::NEG_UNITY_TYPE:
		{
			const NegUnityToken& btok = static_cast<const NegUnityToken&>(b);
			return to_ptr(a - btok);
		}
		case TokenType::NAN_TYPE:
		{
			const NanToken& btok = static_cast<const NanToken&>(b);
			return to_ptr(a - btok);
		}
		case TokenType::NUMBER_TYPE:
		{
			const NumberToken& btok = static_cast<const NumberToken&>(b);
			return to_ptr(a - btok);
		}
		default:
			throw std::runtime_error("Negation can only be applied to tokens Zero, Unity, NegUnity, Nan and Number");
		}
	}

	std::unique_ptr<NumberBaseToken> operator-(const NegUnityToken& a, const Token& b)
	{
		switch (b.get_token_type()) {
		case TokenType::ZERO_TYPE:
		{
			const ZeroToken& btok = static_cast<const ZeroToken&>(b);
			return to_ptr(a - btok);
		}
		case TokenType::UNITY_TYPE:
		{
			const UnityToken& btok = static_cast<const UnityToken&>(b);
			return to_ptr(a - btok);
		}
		case TokenType::NEG_UNITY_TYPE:
		{
			const NegUnityToken& btok = static_cast<const NegUnityToken&>(b);
			return to_ptr(a - btok);
		}
		case TokenType::NAN_TYPE:
		{
			const NanToken& btok = static_cast<const NanToken&>(b);
			return to_ptr(a - btok);
		}
		case TokenType::NUMBER_TYPE:
		{
			const NumberToken& btok = static_cast<const NumberToken&>(b);
			return to_ptr(a - btok);
		}
		default:
			throw std::runtime_error("Negation can only be applied to tokens Zero, Unity, NegUnity, Nan and Number");
		}
	}

	std::unique_ptr<NumberBaseToken> operator-(const NanToken& a, const Token& b)
	{
		switch (b.get_token_type()) {
		case TokenType::ZERO_TYPE:
		{
			const ZeroToken& btok = static_cast<const ZeroToken&>(b);
			return to_ptr(a - btok);
		}
		case TokenType::UNITY_TYPE:
		{
			const UnityToken& btok = static_cast<const UnityToken&>(b);
			return to_ptr(a - btok);
		}
		case TokenType::NEG_UNITY_TYPE:
		{
			const NegUnityToken& btok = static_cast<const NegUnityToken&>(b);
			return to_ptr(a - btok);
		}
		case TokenType::NAN_TYPE:
		{
			const NanToken& btok = static_cast<const NanToken&>(b);
			return to_ptr(a - btok);
		}
		case TokenType::NUMBER_TYPE:
		{
			const NumberToken& btok = static_cast<const NumberToken&>(b);
			return to_ptr(a - btok);
		}
		default:
			throw std::runtime_error("Negation can only be applied to tokens Zero, Unity, NegUnity, Nan and Number");
		}
	}

	std::unique_ptr<NumberBaseToken> operator-(const NumberToken& a, const Token& b)
	{
		switch (b.get_token_type()) {
		case TokenType::ZERO_TYPE:
		{
			const ZeroToken& btok = static_cast<const ZeroToken&>(b);
			return to_ptr(a - btok);
		}
		case TokenType::UNITY_TYPE:
		{
			const UnityToken& btok = static_cast<const UnityToken&>(b);
			return to_ptr(a - btok);
		}
		case TokenType::NEG_UNITY_TYPE:
		{
			const NegUnityToken& btok = static_cast<const NegUnityToken&>(b);
			return to_ptr(a - btok);
		}
		case TokenType::NAN_TYPE:
		{
			const NanToken& btok = static_cast<const NanToken&>(b);
			return to_ptr(a - btok);
		}
		case TokenType::NUMBER_TYPE:
		{
			const NumberToken& btok = static_cast<const NumberToken&>(b);
			return to_ptr(a - btok);
		}
		default:
			throw std::runtime_error("Negation can only be applied to tokens Zero, Unity, NegUnity, Nan and Number");
		}
	}


	//1
	ZeroToken operator-(const ZeroToken& a, const ZeroToken& b)
	{
		return ZeroToken(vc::tc_broadcast_shapes(a.sizes, b.sizes));
	}

	UnityToken operator-(const ZeroToken& a, const NegUnityToken& b)
	{
		return UnityToken(vc::tc_broadcast_shapes(a.sizes, b.sizes));
	}

	NegUnityToken operator-(const ZeroToken& a, const UnityToken& b)
	{
		return NegUnityToken(vc::tc_broadcast_shapes(a.sizes, b.sizes));
	}

	NanToken operator-(const ZeroToken& a, const NanToken& b)
	{
		return NanToken(vc::tc_broadcast_shapes(a.sizes, b.sizes));
	}

	NumberToken operator-(const ZeroToken& a, const NumberToken& b)
	{
		return NumberToken(-b.num, b.is_imaginary, vc::tc_broadcast_shapes(a.sizes, b.sizes));
	}

	// 2
	NegUnityToken operator-(const NegUnityToken& a, const ZeroToken& b)
	{
		return NegUnityToken(vc::tc_broadcast_shapes(a.sizes, b.sizes));
	}

	ZeroToken operator-(const NegUnityToken& a, const NegUnityToken& b)
	{
		return ZeroToken(vc::tc_broadcast_shapes(a.sizes, b.sizes));
	}

	NumberToken operator-(const NegUnityToken& a, const UnityToken& b)
	{
		return NumberToken(-2.0f, false, vc::tc_broadcast_shapes(a.sizes, b.sizes));
	}

	NanToken operator-(const NegUnityToken& a, const NanToken& b)
	{
		return NanToken(vc::tc_broadcast_shapes(a.sizes, b.sizes));
	}

	NumberToken operator-(const NegUnityToken& a, const NumberToken& b)
	{
		return NumberToken(-b.num - 1.0f, b.is_imaginary, vc::tc_broadcast_shapes(a.sizes, b.sizes));
	}

	// 3
	UnityToken operator-(const UnityToken& a, const ZeroToken& b)
	{
		return UnityToken(vc::tc_broadcast_shapes(a.sizes, b.sizes));
	}

	NumberToken operator-(const UnityToken& a, const NegUnityToken& b)
	{
		return NumberToken(2.0f, false, vc::tc_broadcast_shapes(a.sizes, b.sizes));
	}

	ZeroToken operator-(const UnityToken& a, const UnityToken& b)
	{
		return ZeroToken(vc::tc_broadcast_shapes(a.sizes, b.sizes));
	}

	NanToken operator-(const UnityToken& a, const NanToken& b)
	{
		return NanToken(vc::tc_broadcast_shapes(a.sizes, b.sizes));
	}

	NumberToken operator-(const UnityToken& a, const NumberToken& b)
	{
		return NumberToken(1.0f - b.num, b.is_imaginary, vc::tc_broadcast_shapes(a.sizes, b.sizes));
	}

	// 4
	NanToken operator-(const NanToken& a, const ZeroToken& b)
	{
		return NanToken(vc::tc_broadcast_shapes(a.sizes, b.sizes));
	}

	NanToken operator-(const NanToken& a, const NegUnityToken& b)
	{
		return NanToken(vc::tc_broadcast_shapes(a.sizes, b.sizes));
	}

	NanToken operator-(const NanToken& a, const UnityToken& b)
	{
		return NanToken(vc::tc_broadcast_shapes(a.sizes, b.sizes));
	}

	NanToken operator-(const NanToken& a, const NanToken& b)
	{
		return NanToken(vc::tc_broadcast_shapes(a.sizes, b.sizes));
	}

	NanToken operator-(const NanToken& a, const NumberToken& b)
	{
		return NanToken(vc::tc_broadcast_shapes(a.sizes, b.sizes));
	}

	// 5
	NumberToken operator-(const NumberToken& a, const ZeroToken& b)
	{
		return NumberToken(a.num, a.is_imaginary, vc::tc_broadcast_shapes(a.sizes, b.sizes));
	}

	NumberToken operator-(const NumberToken& a, const NegUnityToken& b)
	{
		return NumberToken(a.num + 1.0f, a.is_imaginary, vc::tc_broadcast_shapes(a.sizes, b.sizes));
	}

	NumberToken operator-(const NumberToken& a, const UnityToken& b)
	{
		return NumberToken(a.num - 1.0f, a.is_imaginary, vc::tc_broadcast_shapes(a.sizes, b.sizes));
	}

	NanToken operator-(const NumberToken& a, const NanToken& b)
	{
		return NanToken(vc::tc_broadcast_shapes(a.sizes, b.sizes));
	}

	NumberToken operator-(const NumberToken& a, const NumberToken& b)
	{
		return NumberToken(a.num - b.num, a.is_imaginary || b.is_imaginary, vc::tc_broadcast_shapes(a.sizes, b.sizes));
	}


}