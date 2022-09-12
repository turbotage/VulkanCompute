module;

#include <memory>
#include <stdexcept>
#include <vector>

export module nodes;

import token;

namespace expression {


	std::unique_ptr<NumberBaseToken> copy_token(const Token& tok)
	{
		std::int32_t type = tok.get_token_type();

		switch (type) {
		case TokenType::ZERO_TYPE:
		{
			auto& t = static_cast<const ZeroToken&>(tok);
			return std::make_unique<ZeroToken>(t);
		}
		break;
		case TokenType::UNITY_TYPE:
		{
			auto& t = static_cast<const UnityToken&>(tok);
			return std::make_unique<UnityToken>(t);
		}
		break;
		case TokenType::NEG_UNITY_TYPE:
		{
			auto& t = static_cast<const NegUnityToken&>(tok);
			return std::make_unique<NegUnityToken>(t);
		}
		break;
		case TokenType::NAN_TYPE:
		{
			auto& t = static_cast<const NanToken&>(tok);
			return std::make_unique<NanToken>(t);
		}
		break;
		case TokenType::NUMBER_TYPE:
		{
			auto& t = static_cast<const NumberToken&>(tok);
			return std::make_unique<NumberToken>(t);
		}
		break;
		default:
			throw std::runtime_error("Can't construct TokenNode from token type other than Zero,Unity,NegUnity,Nan,Number");
		}
	}

	export class Node {
	public:

		Node() = default;

		Node(std::unique_ptr<NumberBaseToken> base_token)
			: pToken(std::move(base_token))
		{

		}

		virtual std::string str() = 0;

	public:
		std::vector<std::unique_ptr<Node>> children;
		std::unique_ptr<NumberBaseToken> pToken;
	};

	class TokenNode : public Node {
	public:

		TokenNode(const Token& tok)
			: Node(copy_token(tok))
		{

		}

		std::string str() override {
			
		}

	};

}