module;

#include <memory>
#include <stdexcept>
#include <vector>
#include <string>

#include <symengine/expression.h>
#include <symengine/simplify.h>
#include <symengine/parser.h>

export module nodes;

import lexer;
import token;
import glsl;

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

	class Expression;

	export class Node {
	public:

		Node(Node&&) = default;

		Node(LexContext& ctext)
			: context(ctext)
		{}

		Node(std::vector<std::unique_ptr<Node>>&& childs)
			: context(childs[0]->context)
		{
			children = std::move(childs);
			childs.clear();
		}

		Node(std::vector<std::unique_ptr<Node>>&& childs, LexContext& ctext)
			: children(std::move(childs)), context(ctext)
		{}

		Node(std::unique_ptr<NumberBaseToken> base_token, LexContext& ctext)
			: pToken(std::move(base_token)), context(ctext)
		{}

		virtual std::string str() const = 0;

		virtual std::string glsl_str(const glsl::SymbolicContext& symtext) const = 0;

		virtual std::unique_ptr<Node> copy(LexContext& context) const = 0;

		void fill_variable_list(std::set<std::string>& vars);

		std::unique_ptr<Expression> diff(const std::string& x) const;

		bool child_is_variable(int i) const;

	public:
		LexContext& context;
		std::vector<std::unique_ptr<Node>> children;
		std::unique_ptr<NumberBaseToken> pToken;
	};

	export class TokenNode : public Node {
	public:

		TokenNode(const Token& tok, LexContext& context)
			: Node(copy_token(tok), context)
		{
			
		}

		std::string str() const override {
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
				throw std::runtime_error("Variables should be stored in VariableNodes not TokenNodes");
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

		std::string glsl_str(const glsl::SymbolicContext& symtext) const override {
			return str();
		}

		std::unique_ptr<Node> copy(LexContext& context) const override {
			return std::make_unique<TokenNode>(*pToken, context);
		}

	};

	export class VariableNode : public Node {
	public:

		VariableNode(const VariableToken& token, LexContext& context)
			: Node(context), m_VarToken(token)
		{}

		std::string str() const override {
			return m_VarToken.name;
		}

		std::string glsl_str(const glsl::SymbolicContext& symtext) const override {
			return symtext.get_glsl_var_name(m_VarToken.name);
		}

		std::unique_ptr<Node> copy(LexContext& context) const override {
			return std::make_unique<VariableNode>(m_VarToken, context);
		}

	private:
		VariableToken m_VarToken;
	};

	export std::unique_ptr<Node> node_from_token(const Token& tok, LexContext& context)
	{
		return std::make_unique<TokenNode>(tok, context);
	}

	export class NegNode : public Node {
	public:

		NegNode(std::unique_ptr<Node> child)
			: Node(child->context)
		{
			children.emplace_back(std::move(child));
		}

		std::string str() const override {
			return "(-" + children[0]->str() + ")";
		}

		std::string glsl_str(const glsl::SymbolicContext& symtext) const override {
			return "(-" + children[0]->glsl_str(symtext) + ")";
		}

		std::unique_ptr<Node> copy(LexContext& context) const override {
			return std::make_unique<NegNode>(children[0]->copy(context));
		}

	};

	export class MulNode : public Node {
	public:

		MulNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child)
			: Node(left_child->context)
		{
			children.emplace_back(std::move(left_child));
			children.emplace_back(std::move(right_child));
		}

		std::string str() const override {
			return "(" + children[0]->str() + "*" + children[1]->str() + ")";
		}

		std::string glsl_str(const glsl::SymbolicContext& symtext) const override {
			return "(" + children[0]->glsl_str(symtext) + "*" + children[1]->glsl_str(symtext) + ")";
		}

		std::unique_ptr<Node> copy(LexContext& context) const override {
			return std::make_unique<MulNode>(children[0]->copy(context), children[1]->copy(context));
		}

	};

	export class DivNode : public Node {
	public:

		DivNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child)
			: Node(left_child->context)
		{
			children.emplace_back(std::move(left_child));
			children.emplace_back(std::move(right_child));
		}

		std::string str() const override {
			return "(" + children[0]->str() + "/" + children[1]->str() + ")";
		}

		std::string glsl_str(const glsl::SymbolicContext& symtext) const override {
			return "(" + children[0]->glsl_str(symtext) + "/" + children[1]->glsl_str(symtext) + ")";
		}

		std::unique_ptr<Node> copy(LexContext& context) const override {
			return std::make_unique<DivNode>(children[0]->copy(context), children[1]->copy(context));
		}

	};

	export class AddNode : public Node {
	public:

		AddNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child)
			: Node(left_child->context)
		{
			children.emplace_back(std::move(left_child));
			children.emplace_back(std::move(right_child));
		}

		std::string str() const override {
			return "(" + children[0]->str() + "+" + children[1]->str() + ")";
		}

		std::string glsl_str(const glsl::SymbolicContext& symtext) const override {
			return "(" + children[0]->glsl_str(symtext) + "+" + children[1]->glsl_str(symtext) + ")";
		}

		std::unique_ptr<Node> copy(LexContext& context) const override {
			return std::make_unique<AddNode>(children[0]->copy(context), children[1]->copy(context));
		}

	};

	export class SubNode : public Node {
	public:

		SubNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child)
			: Node(left_child->context)
		{
			children.emplace_back(std::move(left_child));
			children.emplace_back(std::move(right_child));
		}

		std::string str() const override {
			return "(" + children[0]->str() + "-" + children[1]->str() + ")";
		}

		std::string glsl_str(const glsl::SymbolicContext& symtext) const override {
			return "(" + children[0]->glsl_str(symtext) + "-" + children[1]->glsl_str(symtext) + ")";
		}

		std::unique_ptr<Node> copy(LexContext& context) const override {
			return std::make_unique<SubNode>(children[0]->copy(context), children[1]->copy(context));
		}

	};

	export class PowNode : public Node {
	public:

		PowNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child)
			: Node(left_child->context)
		{
			children.emplace_back(std::move(left_child));
			children.emplace_back(std::move(right_child));
		}

		std::string str() const override {
			return "pow(" + children[0]->str() + "," + children[1]->str() + ")";
		}
		
		std::string glsl_str(const glsl::SymbolicContext& symtext) const override {
			return "pow(" + children[0]->glsl_str(symtext) + "," + children[1]->glsl_str(symtext) + ")";
		}

		std::unique_ptr<Node> copy(LexContext& context) const override {
			return std::make_unique<PowNode>(children[0]->copy(context), children[1]->copy(context));
		}

	};

	// UNARY

	export class SgnNode : public Node {
	public:

		SgnNode(std::unique_ptr<Node> child)
			: Node(child->context)
		{
			children.emplace_back(std::move(child));
		}

		std::string str() const override {
			return "sgn(" + children[0]->str() + ")";
		}

		std::string glsl_str(const glsl::SymbolicContext& symtext) const override {
			return "sign(" + children[0]->glsl_str(symtext) + ")";
		}

		std::unique_ptr<Node> copy(LexContext& context) const override {
			return std::make_unique<SgnNode>(children[0]->copy(context));
		}

	};

	export class AbsNode : public Node {
	public:

		AbsNode(std::unique_ptr<Node> child)
			: Node(child->context)
		{
			children.emplace_back(std::move(child));
		}

		std::string str() const override {
			return "abs(" + children[0]->str() + ")";
		}

		std::string glsl_str(const glsl::SymbolicContext& symtext) const override {
			return "abs(" + children[0]->glsl_str(symtext) + ")";
		}

		std::unique_ptr<Node> copy(LexContext& context) const override {
			return std::make_unique<AbsNode>(children[0]->copy(context));
		}

	};

	export class SqrtNode : public Node {
	public:

		SqrtNode(std::unique_ptr<Node> child)
			: Node(child->context)
		{
			children.emplace_back(std::move(child));
		}

		std::string str() const override {
			return "sqrt(" + children[0]->str() + ")";
		}

		std::string glsl_str(const glsl::SymbolicContext& symtext) const override {
			return "sqrt(" + children[0]->glsl_str(symtext) + ")";
		}

		std::unique_ptr<Node> copy(LexContext& context) const override {
			return std::make_unique<SqrtNode>(children[0]->copy(context));
		}

	};

	export class ExpNode : public Node {
	public:

		ExpNode(std::unique_ptr<Node> child)
			: Node(child->context)
		{
			children.emplace_back(std::move(child));
		}

		std::string str() const override {
			return "exp(" + children[0]->str() + ")";
		}

		std::string glsl_str(const glsl::SymbolicContext& symtext) const override {
			return "exp(" + children[0]->glsl_str(symtext) + ")";
		}

		std::unique_ptr<Node> copy(LexContext& context) const override {
			return std::make_unique<ExpNode>(children[0]->copy(context));
		}

	};

	export class LogNode : public Node {
	public:

		LogNode(std::unique_ptr<Node> child)
			: Node(child->context)
		{
			children.emplace_back(std::move(child));
		}

		std::string str() const override {
			return "log(" + children[0]->str() + ")";
		}

		std::string glsl_str(const glsl::SymbolicContext& symtext) const override {
			return "log(" + children[0]->glsl_str(symtext) + ")";
		}

		std::unique_ptr<Node> copy(LexContext& context) const override {
			return std::make_unique<LogNode>(children[0]->copy(context));
		}

	};

	export class SinNode : public Node {
	public:

		SinNode(std::unique_ptr<Node> child)
			: Node(child->context)
		{
			children.emplace_back(std::move(child));
		}

		std::string str() const override {
			return "sin(" + children[0]->str() + ")";
		}

		std::string glsl_str(const glsl::SymbolicContext& symtext) const override {
			return "sin(" + children[0]->glsl_str(symtext) + ")";
		}

		std::unique_ptr<Node> copy(LexContext& context) const override {
			return std::make_unique<SinNode>(children[0]->copy(context));
		}

	};

	export class CosNode : public Node {
	public:

		CosNode(std::unique_ptr<Node> child)
			: Node(child->context)
		{
			children.emplace_back(std::move(child));
		}

		std::string str() const override {
			return "cos(" + children[0]->str() + ")";
		}

		std::string glsl_str(const glsl::SymbolicContext& symtext) const override {
			return "cos(" + children[0]->glsl_str(symtext) + ")";
		}

		std::unique_ptr<Node> copy(LexContext& context) const override {
			return std::make_unique<CosNode>(children[0]->copy(context));
		}

	};

	export class TanNode : public Node {
	public:

		TanNode(std::unique_ptr<Node> child)
			: Node(child->context)
		{
			children.emplace_back(std::move(child));
		}

		std::string str() const override {
			return "tan(" + children[0]->str() + ")";
		}

		std::string glsl_str(const glsl::SymbolicContext& symtext) const override {
			return "tan(" + children[0]->glsl_str(symtext) + ")";
		}

		std::unique_ptr<Node> copy(LexContext& context) const override {
			return std::make_unique<TanNode>(children[0]->copy(context));
		}

	};

	export class AsinNode : public Node {
	public:

		AsinNode(std::unique_ptr<Node> child)
			: Node(child->context)
		{
			children.emplace_back(std::move(child));
		}

		std::string str() const override {
			return "asin(" + children[0]->str() + ")";
		}

		std::string glsl_str(const glsl::SymbolicContext& symtext) const override {
			return "asin(" + children[0]->glsl_str(symtext) + ")";
		}

		std::unique_ptr<Node> copy(LexContext& context) const override {
			return std::make_unique<AsinNode>(children[0]->copy(context));
		}

	};

	export class AcosNode : public Node {
	public:

		AcosNode(std::unique_ptr<Node> child)
			: Node(child->context)
		{
			children.emplace_back(std::move(child));
		}

		std::string str() const override {
			return "acos(" + children[0]->str() + ")";
		}

		std::string glsl_str(const glsl::SymbolicContext& symtext) const override {
			return "acos(" + children[0]->glsl_str(symtext) + ")";
		}

		std::unique_ptr<Node> copy(LexContext& context) const override {
			return std::make_unique<AcosNode>(children[0]->copy(context));
		}

	};

	export class AtanNode : public Node {
	public:

		AtanNode(std::unique_ptr<Node> child)
			: Node(child->context)
		{
			children.emplace_back(std::move(child));
		}

		std::string str() const override {
			return "atan(" + children[0]->str() + ")";
		}

		std::string glsl_str(const glsl::SymbolicContext& symtext) const override {
			return "atan(" + children[0]->glsl_str(symtext) + ")";
		}

		std::unique_ptr<Node> copy(LexContext& context) const override {
			return std::make_unique<AtanNode>(children[0]->copy(context));
		}

	};

	export class SinhNode : public Node {
	public:

		SinhNode(std::unique_ptr<Node> child)
			: Node(child->context)
		{
			children.emplace_back(std::move(child));
		}

		std::string str() const override {
			return "sinh(" + children[0]->str() + ")";
		}

		std::string glsl_str(const glsl::SymbolicContext& symtext) const override {
			return "sinh(" + children[0]->glsl_str(symtext) + ")";
		}

		std::unique_ptr<Node> copy(LexContext& context) const override {
			return std::make_unique<SinhNode>(children[0]->copy(context));
		}

	};

	export class CoshNode : public Node {
	public:

		CoshNode(std::unique_ptr<Node> child)
			: Node(child->context)
		{
			children.emplace_back(std::move(child));
		}

		std::string str() const override {
			return "cosh(" + children[0]->str() + ")";
		}

		std::string glsl_str(const glsl::SymbolicContext& symtext) const override {
			return "cosh(" + children[0]->glsl_str(symtext) + ")";
		}

		std::unique_ptr<Node> copy(LexContext& context) const override {
			return std::make_unique<CoshNode>(children[0]->copy(context));
		}

	};

	export class TanhNode : public Node {
	public:

		TanhNode(std::unique_ptr<Node> child)
			: Node(child->context)
		{
			children.emplace_back(std::move(child));
		}

		std::string str() const override {
			return "tanh(" + children[0]->str() + ")";
		}

		std::string glsl_str(const glsl::SymbolicContext& symtext) const override {
			return "tanh(" + children[0]->glsl_str(symtext) + ")";
		}

		std::unique_ptr<Node> copy(LexContext& context) const override {
			return std::make_unique<TanhNode>(children[0]->copy(context));
		}

	};

	export class AsinhNode : public Node {
	public:

		AsinhNode(std::unique_ptr<Node> child)
			: Node(child->context)
		{
			children.emplace_back(std::move(child));
		}

		std::string str() const override {
			return "asinh(" + children[0]->str() + ")";
		}

		std::string glsl_str(const glsl::SymbolicContext& symtext) const override {
			return "asinh(" + children[0]->glsl_str(symtext) + ")";
		}

		std::unique_ptr<Node> copy(LexContext& context) const override {
			return std::make_unique<AsinhNode>(children[0]->copy(context));
		}

	};

	export class AcoshNode : public Node {
	public:

		AcoshNode(std::unique_ptr<Node> child)
			: Node(child->context)
		{
			children.emplace_back(std::move(child));
		}

		std::string str() const override {
			return "acosh(" + children[0]->str() + ")";
		}

		std::string glsl_str(const glsl::SymbolicContext& symtext) const override {
			return "acosh(" + children[0]->glsl_str(symtext) + ")";
		}

		std::unique_ptr<Node> copy(LexContext& context) const override {
			return std::make_unique<AcoshNode>(children[0]->copy(context));
		}

	};

	export class AtanhNode : public Node {
	public:

		AtanhNode(std::unique_ptr<Node> child)
			: Node(child->context)
		{
			children.emplace_back(std::move(child));
		}

		std::string str() const override {
			return "atanh(" + children[0]->str() + ")";
		}

		std::string glsl_str(const glsl::SymbolicContext& symtext) const override {
			return "atanh(" + children[0]->glsl_str(symtext) + ")";
		}

		std::unique_ptr<Node> copy(LexContext& context) const override {
			return std::make_unique<AtanhNode>(children[0]->copy(context));
		}

	};

	export class DerivativeNode : public Node {
	public:

		DerivativeNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child);

		std::string str() const override {
			return children[0]->str();
		}

		std::string glsl_str(const glsl::SymbolicContext& symtext) const override {
			return children[0]->glsl_str(symtext);
		}

		std::unique_ptr<Node> copy(LexContext& context) const override {
			return std::make_unique<DerivativeNode>(children[0]->copy(context), children[1]->copy(context));
		}

	private:

	};

	export class SubsNode : public Node {
	public:

		SubsNode(std::vector<std::unique_ptr<Node>>&& childs);

		std::string str() const override;

		std::string glsl_str(const glsl::SymbolicContext& symtext) const override;

		std::unique_ptr<Node> copy(LexContext& context) const override {
			std::vector<std::unique_ptr<Node>> copied_children;
			copied_children.reserve(children.size());

			for (auto& child : children) {
				copied_children.push_back(std::move(child->copy(context)));
			}

			return std::make_unique<SubsNode>(std::move(copied_children));
		}

	private:

	};

}