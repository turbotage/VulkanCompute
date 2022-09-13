module;

#include <memory>
#include <stdexcept>
#include <vector>
/*
#include <symengine/expression.h>
#include <symengine/simplify.h>
#include <symengine/parser.h>
*/

export module nodes;

import lexer;
import token;

namespace expression {

	export std::unique_ptr<NumberBaseToken> copy_token(const Token& tok)
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

		Node(const LexContext& context)
			: context(context)
		{}

		Node(std::unique_ptr<NumberBaseToken> base_token, const LexContext& context)
			: pToken(std::move(base_token)), context(context)
		{

		}

		virtual std::string str() = 0;

	public:
		const LexContext& context;
		std::vector<std::unique_ptr<Node>> children;
		std::unique_ptr<NumberBaseToken> pToken;
	};

	export class TokenNode : public Node {
	public:

		TokenNode(const Token& tok, const LexContext& context)
			: Node(copy_token(tok), context)
		{

		}

		std::string str() override {
			switch (pToken->get_token_type()) {
			case TokenType::NO_TOKEN_TYPE:
				throw std::runtime_error("An expression graph cannot contain a NO_TOKEN_TYPE token");
			case TokenType::OPERATOR_TYPE:
				throw std::runtime_error("Operator type must be more speciallized, unary or binary?");
				break;
			case TokenType::UNARY_OPERATOR_TYPE:
				{
					auto id = pToken->get_id();
					return context.operator_id_name_map.at(id);
				}
			case TokenType::BINARY_OPERATOR_TYPE:
				{
					auto id = pToken->get_id();
					return context.operator_id_name_map.at(id);
				}
			case TokenType::FUNCTION_TYPE:
				{
					auto id = pToken->get_id();
					return context.function_id_name_map.at(id);
				}
			case TokenType::VARIABLE_TYPE:
				{
					const VariableToken& vtok = dynamic_cast<const VariableToken&>(*pToken);
					return vtok.name;
				}
			case TokenType::NUMBER_TYPE:
				{
					const NumberToken& ntok = dynamic_cast<const NumberToken&>(*pToken);
					return ntok.name;
				}
			case TokenType::ZERO_TYPE:
				return "0";
			case TokenType::UNITY_TYPE:
				return "1";
			case TokenType::NEG_UNITY_TYPE:
				return "-1";
			case TokenType::NAN_TYPE:
				return "NaN";
			case TokenType::LEFT_PAREN_TYPE:
				return "(";
			case TokenType::RIGHT_PAREN_TYPE:
				return ")";
			case TokenType::COMMA_TYPE:
				return ",";
			default:
				throw std::runtime_error("Unexpected token type");
			}
		}

	};

	export class NegNode : public Node {
	public:

		NegNode(std::unique_ptr<Node> child)
			: Node(child->context)
		{
			children = { std::move(child) };
		}

		std::string str() override {
			return "(-" + children[0]->str() + ")";
		}

	};

	export class MulNode : public Node {
	public:

		MulNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child)
			: Node(left_child->context)
		{
			children = { std::move(left_child), std::move(right_child) };
		}

		std::string str() override {
			return "(" + children[0]->str() + "*" + children[1]->str() + ")";
		}
	};

	export class DivNode : public Node {
	public:

		DivNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child)
			: Node(left_child->context)
		{
			children = { std::move(left_child), std::move(right_child) };
		}

		std::string str() override {
			return "(" + children[0]->str() + "/" + children[1]->str() + ")";
		}

	};

	export class AddNode : public Node {
	public:

		AddNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child)
			: Node(left_child->context)
		{
			children = { std::move(left_child), std::move(right_child) };
		}

		std::string str() override {
			return "(" + children[0]->str() + "+" + children[1]->str() + ")";
		}

	};

	export class SubNode : public Node {
	public:

		SubNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child)
			: Node(left_child->context)
		{
			children = { std::move(left_child), std::move(right_child) };
		}

		std::string str() override {
			return "(" + children[0]->str() + "-" + children[1]->str() + ")";
		}

	};

	export class PowNode : public Node {
	public:

		PowNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child)
			: Node(left_child->context)
		{
			children = { std::move(left_child), std::move(right_child) };
		}

		std::string str() override {
			return "pow(" + children[0]->str() + "," + children[1]->str() + ")";
		}

	};

	// UNARY

	export class SgnNode : public Node {
	public:

		SgnNode(std::unique_ptr<Node> child)
			: Node(child->context)
		{
			children = { std::move(child) };
		}
		
		std::string str() override {
			return "sign(" + children[0]->str() + ")";
		}

	};

	export class AbsNode : public Node {
	public:

		AbsNode(std::unique_ptr<Node> child)
			: Node(child->context)
		{
			children = { std::move(child) };
		}

		std::string str() override {
			return "abs(" + children[0]->str() + ")";
		}

	};

	export class SqrtNode : public Node {
	public:

		SqrtNode(std::unique_ptr<Node> child)
			: Node(child->context)
		{
			children = { std::move(child) };
		}

		std::string str() override {
			return "sqrt(" + children[0]->str() + ")";
		}

	};

	export class ExpNode : public Node {
	public:

		ExpNode(std::unique_ptr<Node> child)
			: Node(child->context)
		{
			children = { std::move(child) };
		}

		std::string str() override {
			return "exp(" + children[0]->str() + ")";
		}

	};

	export class LogNode : public Node {
	public:

		LogNode(std::unique_ptr<Node> child)
			: Node(child->context)
		{
			children = { std::move(child) };
		}

		std::string str() override {
			return "log(" + children[0]->str() + ")";
		}

	};

	export class SinNode : public Node {
	public:

		SinNode(std::unique_ptr<Node> child)
			: Node(child->context)
		{
			children = { std::move(child) };
		}

		std::string str() override {
			return "sin(" + children[0]->str() + ")";
		}

	};

	export class CosNode : public Node {
	public:

		CosNode(std::unique_ptr<Node> child)
			: Node(child->context)
		{
			children = { std::move(child) };
		}

		std::string str() override {
			return "cos(" + children[0]->str() + ")";
		}

	};

	export class TanNode : public Node {
	public:

		TanNode(std::unique_ptr<Node> child)
			: Node(child->context)
		{
			children = { std::move(child) };
		}

		std::string str() override {
			return "tan(" + children[0]->str() + ")";
		}

	};

	export class AsinNode : public Node {
	public:

		AsinNode(std::unique_ptr<Node> child)
			: Node(child->context)
		{
			children = { std::move(child) };
		}

		std::string str() override {
			return "asin(" + children[0]->str() + ")";
		}

	};

	export class AcosNode : public Node {
	public:

		AcosNode(std::unique_ptr<Node> child)
			: Node(child->context)
		{
			children = { std::move(child) };
		}

		std::string str() override {
			return "acos(" + children[0]->str() + ")";
		}

	};

	export class AtanNode : public Node {
	public:

		AtanNode(std::unique_ptr<Node> child)
			: Node(child->context)
		{
			children = { std::move(child) };
		}

		std::string str() override {
			return "atan(" + children[0]->str() + ")";
		}

	};

	export class SinhNode : public Node {
	public:

		SinhNode(std::unique_ptr<Node> child)
			: Node(child->context)
		{
			children = { std::move(child) };
		}

		std::string str() override {
			return "sinh(" + children[0]->str() + ")";
		}

	};

	export class CoshNode : public Node {
	public:

		CoshNode(std::unique_ptr<Node> child)
			: Node(child->context)
		{
			children = { std::move(child) };
		}

		std::string str() override {
			return "cosh(" + children[0]->str() + ")";
		}

	};

	export class TanhNode : public Node {
	public:

		TanhNode(std::unique_ptr<Node> child)
			: Node(child->context)
		{
			children = { std::move(child) };
		}

		std::string str() override {
			return "tanh(" + children[0]->str() + ")";
		}

	};

	export class AsinhNode : public Node {
	public:

		AsinhNode(std::unique_ptr<Node> child)
			: Node(child->context)
		{
			children = { std::move(child) };
		}

		std::string str() override {
			return "asinh(" + children[0]->str() + ")";
		}

	};

	export class AcoshNode : public Node {
	public:

		AcoshNode(std::unique_ptr<Node> child)
			: Node(child->context)
		{
			children = { std::move(child) };
		}

		std::string str() override {
			return "acosh(" + children[0]->str() + ")";
		}

	};

	export class AtanhNode : public Node {
	public:

		AtanhNode(std::unique_ptr<Node> child)
			: Node(child->context)
		{
			children = { std::move(child) };
		}

		std::string str() override {
			return "atanh(" + children[0]->str() + ")";
		}

	};

	export class DerivativeNode : public Node {
	public:

		DerivativeNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child)
			: Node(left_child->context)
		{
			children = { std::move(left_child), std::move(right_child) };
		}

		std::string str() override {
			auto child_ptr = dynamic_cast<const AbsNode*>(children[0].get());
			if (child_ptr != nullptr) {
				auto& under_abs_child = child_ptr->children[0];
				std::string rstr = "sign(" + under_abs_child->str() + ")*(";
				//rstr += SymEngine::parse(under_abs_child->str())->diff(SymEngine::symbol(children[1]->str()))->__str__() + ")";
				return rstr;
			}

			return "";
		}

	};

}