module;

#include <symengine/expression.h>
#include <symengine/simplify.h>
#include <symengine/parser.h>

module expr;

import <memory>;
import <deque>;
import <unordered_map>;
import <initializer_list>;
import <functional>;
import <stdexcept>;
import <vector>;
import <tuple>;
import <algorithm>;
import <set>;
import <iterator>;
import <optional>;

import symbolic;
import shunter;
import defaultexp;

using namespace expression;

void symengine_get_args(const SymEngine::RCP<const SymEngine::Basic>& subexpr, std::set<std::string>& args)
{
	auto vec_args = subexpr->get_args();

	if (vec_args.size() == 0) {
		if (SymEngine::is_a_Number(*subexpr)) {
			return;
		}
		else if (SymEngine::is_a<SymEngine::FunctionSymbol>(*subexpr)) {
			return;
		}
		else if (SymEngine::is_a<SymEngine::Constant>(*subexpr)) {
			return;
		}

		args.insert(subexpr->__str__());
	}
	else {
		for (auto& varg : vec_args) {
			symengine_get_args(varg, args);
		}
	}
}

std::string symengine_parse(const std::string& to_parse)
{
	auto p = SymEngine::parse(to_parse);
	return p->__str__();
}

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

// NODE

Node::Node(LexContext& ctext)
	: context(ctext)
{}

Node::Node(std::vector<std::unique_ptr<Node>>&& childs)
	: context(childs[0]->context)
{
	children = std::move(childs);
	childs.clear();
}

Node::Node(std::vector<std::unique_ptr<Node>>&& childs, LexContext& ctext)
	: children(std::move(childs)), context(ctext)
{}

Node::Node(std::unique_ptr<NumberBaseToken> base_token, LexContext& ctext)
	: pToken(std::move(base_token)), context(ctext)
{}

void Node::fill_variable_list(std::set<std::string>& vars)
{
	VariableNode* var_node = dynamic_cast<VariableNode*>(this);
	if (var_node != nullptr) {
		vars.insert(var_node->str());
	}

	for (auto& child : children) {
		child->fill_variable_list(vars);
	}
}

std::unique_ptr<Expression> Node::diff(const std::string& x) const
{
	std::string expr_str = util::to_lower_case(util::remove_whitespace(str()));

	LexContext new_context(context);

	SymEngine::Assumptions assum1(new_context.variable_assumptions);

	auto parsed = SymEngine::simplify(SymEngine::parse(expr_str), &assum1);
	auto dexpr = parsed->diff(SymEngine::symbol(
		util::to_lower_case(
			util::remove_whitespace(x))));

	std::set<std::string> vars;
	symengine_get_args(dexpr, vars);

	new_context.variables.reserve(vars.size());
	for (auto& var : vars) {
		if (!util::container_contains(new_context.variables, var)) {
			new_context.variables.emplace_back(var);
			new_context.variable_assumptions.insert(SymEngine::contains(SymEngine::symbol(var), SymEngine::reals()));
		}
	}

	SymEngine::Assumptions assum2(new_context.variable_assumptions);

	auto sim_dexpr = SymEngine::simplify(dexpr, &assum2);

	auto dexpr_str = util::to_lower_case(
		util::remove_whitespace(sim_dexpr->__str__()));

	return std::make_unique<Expression>(dexpr_str, new_context);
}

bool expression::Node::child_is_variable(int i) const
{
	const VariableNode* var_node = dynamic_cast<const VariableNode*>(children.at(i).get());
	return var_node != nullptr;
}

std::unique_ptr<Node> node_from_token(const Token& tok, LexContext& context)
{
	return std::make_unique<TokenNode>(tok, context);
}

// TOKEN NODE

TokenNode::TokenNode(const Token& tok, LexContext& context)
	: Node(copy_token(tok), context)
{}

std::string TokenNode::str() const 
{
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

std::string TokenNode::glsl_str(const glsl::SymbolicContext& symtext) const
{
	return str();
}

std::unique_ptr<Node> TokenNode::copy(LexContext& context) const
{
	return std::make_unique<TokenNode>(*pToken, context);
}

// VARIABLE NODE

VariableNode::VariableNode(const VariableToken& token, LexContext& context)
	: Node(context), m_VarToken(token)
{}

std::string VariableNode::str() const
{
	return m_VarToken.name;
}

std::string VariableNode::glsl_str(const glsl::SymbolicContext& symtext) const
{
	return symtext.get_glsl_var_name(m_VarToken.name);
}

std::unique_ptr<Node> VariableNode::copy(LexContext& context) const
{
	return std::make_unique<VariableNode>(m_VarToken, context);
}

// NEG NODE

NegNode::NegNode(std::unique_ptr<Node> child)
	: Node(child->context)
{
	children.emplace_back(std::move(child));
}

std::string NegNode::str() const
{
	return "(-" + children[0]->str() + ")";
}

std::string NegNode::glsl_str(const glsl::SymbolicContext& symtext) const
{
	return "(-" + children[0]->glsl_str(symtext) + ")";
}

std::unique_ptr<Node> NegNode::copy(LexContext& context) const
{
	return std::make_unique<NegNode>(children[0]->copy(context));
}

// MUL NODE
MulNode::MulNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child)
	: Node(left_child->context)
{
	children.emplace_back(std::move(left_child));
	children.emplace_back(std::move(right_child));
}

std::string MulNode::str() const
{
	return "(" + children[0]->str() + "*" + children[1]->str() + ")";
}

std::string MulNode::glsl_str(const glsl::SymbolicContext& symtext) const
{
	return "(" + children[0]->glsl_str(symtext) + "*" + children[1]->glsl_str(symtext) + ")";
}

std::unique_ptr<Node> MulNode::copy(LexContext& context) const
{
	return std::make_unique<MulNode>(children[0]->copy(context), children[1]->copy(context));
}

// DIV NODE
DivNode::DivNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child)
	: Node(left_child->context)
{
	children.emplace_back(std::move(left_child));
	children.emplace_back(std::move(right_child));
}

std::string DivNode::str() const
{
	return "(" + children[0]->str() + "/" + children[1]->str() + ")";
}

std::string DivNode::glsl_str(const glsl::SymbolicContext& symtext) const
{
	return "(" + children[0]->glsl_str(symtext) + "/" + children[1]->glsl_str(symtext) + ")";
}

std::unique_ptr<Node> DivNode::copy(LexContext& context) const
{
	return std::make_unique<DivNode>(children[0]->copy(context), children[1]->copy(context));
}

// ADD NODE
AddNode::AddNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child)
	: Node(left_child->context)
{
	children.emplace_back(std::move(left_child));
	children.emplace_back(std::move(right_child));
}

std::string AddNode::str() const
{
	return "(" + children[0]->str() + "+" + children[1]->str() + ")";
}

std::string AddNode::glsl_str(const glsl::SymbolicContext& symtext) const
{
	return "(" + children[0]->glsl_str(symtext) + "+" + children[1]->glsl_str(symtext) + ")";
}

std::unique_ptr<Node> AddNode::copy(LexContext& context) const
{
	return std::make_unique<AddNode>(children[0]->copy(context), children[1]->copy(context));
}

// SUB NODE
SubNode::SubNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child)
	: Node(left_child->context)
{
	children.emplace_back(std::move(left_child));
	children.emplace_back(std::move(right_child));
}

std::string SubNode::str() const
{
	return "(" + children[0]->str() + "-" + children[1]->str() + ")";
}

std::string SubNode::glsl_str(const glsl::SymbolicContext& symtext) const
{
	return "(" + children[0]->glsl_str(symtext) + "-" + children[1]->glsl_str(symtext) + ")";
}

std::unique_ptr<Node> SubNode::copy(LexContext& context) const
{
	return std::make_unique<SubNode>(children[0]->copy(context), children[1]->copy(context));
}

// POW NODE
PowNode::PowNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child)
	: Node(left_child->context)
{
	children.emplace_back(std::move(left_child));
	children.emplace_back(std::move(right_child));
}

std::string PowNode::str() const
{
	return "pow(" + children[0]->str() + "," + children[1]->str() + ")";
}

std::string PowNode::glsl_str(const glsl::SymbolicContext& symtext) const
{
	return "pow(" + children[0]->glsl_str(symtext) + "," + children[1]->glsl_str(symtext) + ")";
}

std::unique_ptr<Node> PowNode::copy(LexContext& context) const
{
	return std::make_unique<PowNode>(children[0]->copy(context), children[1]->copy(context));
}

// SGN NODE
SgnNode::SgnNode(std::unique_ptr<Node> child)
	: Node(child->context)
{
	children.emplace_back(std::move(child));
}

std::string SgnNode::str() const
{
	return "sgn(" + children[0]->str() + ")";
}

std::string SgnNode::glsl_str(const glsl::SymbolicContext& symtext) const
{
	return "sign(" + children[0]->glsl_str(symtext) + ")";
}

std::unique_ptr<Node> SgnNode::copy(LexContext& context) const
{
	return std::make_unique<SgnNode>(children[0]->copy(context));
}

// ABS NODE
AbsNode::AbsNode(std::unique_ptr<Node> child)
	: Node(child->context)
{
	children.emplace_back(std::move(child));
}

std::string AbsNode::str() const
{
	return "abs(" + children[0]->str() + ")";
}

std::string AbsNode::glsl_str(const glsl::SymbolicContext& symtext) const
{
	return "abs(" + children[0]->glsl_str(symtext) + ")";
}

std::unique_ptr<Node> AbsNode::copy(LexContext& context) const
{
	return std::make_unique<AbsNode>(children[0]->copy(context));
}

// SQRT NODE
SqrtNode::SqrtNode(std::unique_ptr<Node> child)
	: Node(child->context)
{
	children.emplace_back(std::move(child));
}

std::string SqrtNode::str() const
{
	return "sqrt(" + children[0]->str() + ")";
}

std::string SqrtNode::glsl_str(const glsl::SymbolicContext& symtext) const
{
	return "sqrt(" + children[0]->glsl_str(symtext) + ")";
}

std::unique_ptr<Node> SqrtNode::copy(LexContext& context) const
{
	return std::make_unique<SqrtNode>(children[0]->copy(context));
}

// EXP NODE
ExpNode::ExpNode(std::unique_ptr<Node> child)
	: Node(child->context)
{
	children.emplace_back(std::move(child));
}

std::string ExpNode::str() const
{
	return "exp(" + children[0]->str() + ")";
}

std::string ExpNode::glsl_str(const glsl::SymbolicContext& symtext) const
{
	return "exp(" + children[0]->glsl_str(symtext) + ")";
}

std::unique_ptr<Node> ExpNode::copy(LexContext& context) const
{
	return std::make_unique<ExpNode>(children[0]->copy(context));
}

// LOG NODE
LogNode::LogNode(std::unique_ptr<Node> child)
	: Node(child->context)
{
	children.emplace_back(std::move(child));
}

std::string LogNode::str() const
{
	return "log(" + children[0]->str() + ")";
}

std::string LogNode::glsl_str(const glsl::SymbolicContext& symtext) const
{
	return "log(" + children[0]->glsl_str(symtext) + ")";
}

std::unique_ptr<Node> LogNode::copy(LexContext& context) const
{
	return std::make_unique<LogNode>(children[0]->copy(context));
}

// SIN NODE
SinNode::SinNode(std::unique_ptr<Node> child)
	: Node(child->context)
{
	children.emplace_back(std::move(child));
}

std::string SinNode::str() const
{
	return "sin(" + children[0]->str() + ")";
}

std::string SinNode::glsl_str(const glsl::SymbolicContext& symtext) const
{
	return "sin(" + children[0]->glsl_str(symtext) + ")";
}

std::unique_ptr<Node> SinNode::copy(LexContext& context) const
{
	return std::make_unique<SinNode>(children[0]->copy(context));
}

// COS NODE
CosNode::CosNode(std::unique_ptr<Node> child)
	: Node(child->context)
{
	children.emplace_back(std::move(child));
}

std::string CosNode::str() const
{
	return "cos(" + children[0]->str() + ")";
}

std::string CosNode::glsl_str(const glsl::SymbolicContext& symtext) const
{
	return "cos(" + children[0]->glsl_str(symtext) + ")";
}

std::unique_ptr<Node> CosNode::copy(LexContext& context) const
{
	return std::make_unique<CosNode>(children[0]->copy(context));
}

// TAN NODE
TanNode::TanNode(std::unique_ptr<Node> child)
	: Node(child->context)
{
	children.emplace_back(std::move(child));
}

std::string TanNode::str() const
{
	return "tan(" + children[0]->str() + ")";
}

std::string TanNode::glsl_str(const glsl::SymbolicContext& symtext) const
{
	return "tan(" + children[0]->glsl_str(symtext) + ")";
}

std::unique_ptr<Node> TanNode::copy(LexContext& context) const
{
	return std::make_unique<TanNode>(children[0]->copy(context));
}

// ASIN NODE
AsinNode::AsinNode(std::unique_ptr<Node> child)
	: Node(child->context)
{
	children.emplace_back(std::move(child));
}

std::string AsinNode::str() const
{
	return "asin(" + children[0]->str() + ")";
}

std::string AsinNode::glsl_str(const glsl::SymbolicContext& symtext) const
{
	return "asin(" + children[0]->glsl_str(symtext) + ")";
}

std::unique_ptr<Node> AsinNode::copy(LexContext& context) const
{
	return std::make_unique<AsinNode>(children[0]->copy(context));
}

// ACOS NODE
AcosNode::AcosNode(std::unique_ptr<Node> child)
	: Node(child->context)
{
	children.emplace_back(std::move(child));
}

std::string AcosNode::str() const
{
	return "acos(" + children[0]->str() + ")";
}

std::string AcosNode::glsl_str(const glsl::SymbolicContext& symtext) const
{
	return "acos(" + children[0]->glsl_str(symtext) + ")";
}

std::unique_ptr<Node> AcosNode::copy(LexContext& context) const
{
	return std::make_unique<AcosNode>(children[0]->copy(context));
}

// ATAN NODE
AtanNode::AtanNode(std::unique_ptr<Node> child)
	: Node(child->context)
{
	children.emplace_back(std::move(child));
}

std::string AtanNode::str() const
{
	return "atan(" + children[0]->str() + ")";
}

std::string AtanNode::glsl_str(const glsl::SymbolicContext& symtext) const
{
	return "atan(" + children[0]->glsl_str(symtext) + ")";
}

std::unique_ptr<Node> AtanNode::copy(LexContext& context) const
{
	return std::make_unique<AtanNode>(children[0]->copy(context));
}

// SINH NODE
SinhNode::SinhNode(std::unique_ptr<Node> child)
	: Node(child->context)
{
	children.emplace_back(std::move(child));
}

std::string SinhNode::str() const
{
	return "sinh(" + children[0]->str() + ")";
}

std::string SinhNode::glsl_str(const glsl::SymbolicContext& symtext) const
{
	return "sinh(" + children[0]->glsl_str(symtext) + ")";
}

std::unique_ptr<Node> SinhNode::copy(LexContext& context) const
{
	return std::make_unique<SinhNode>(children[0]->copy(context));
}

// COSH NODE
CoshNode::CoshNode(std::unique_ptr<Node> child)
	: Node(child->context)
{
	children.emplace_back(std::move(child));
}

std::string CoshNode::str() const
{
	return "cosh(" + children[0]->str() + ")";
}

std::string CoshNode::glsl_str(const glsl::SymbolicContext& symtext) const
{
	return "cosh(" + children[0]->glsl_str(symtext) + ")";
}

std::unique_ptr<Node> CoshNode::copy(LexContext& context) const
{
	return std::make_unique<CoshNode>(children[0]->copy(context));
}

// TANH NODE
TanhNode::TanhNode(std::unique_ptr<Node> child)
	: Node(child->context)
{
	children.emplace_back(std::move(child));
}

std::string TanhNode::str() const
{
	return "tanh(" + children[0]->str() + ")";
}

std::string TanhNode::glsl_str(const glsl::SymbolicContext& symtext) const
{
	return "tanh(" + children[0]->glsl_str(symtext) + ")";
}

std::unique_ptr<Node> TanhNode::copy(LexContext& context) const
{
	return std::make_unique<TanhNode>(children[0]->copy(context));
}

// ASINH NODE
AsinhNode::AsinhNode(std::unique_ptr<Node> child)
: Node(child->context)
{
	children.emplace_back(std::move(child));
}

std::string AsinhNode::str() const
{
	return "asinh(" + children[0]->str() + ")";
}

std::string AsinhNode::glsl_str(const glsl::SymbolicContext& symtext) const
{
	return "asinh(" + children[0]->glsl_str(symtext) + ")";
}

std::unique_ptr<Node> AsinhNode::copy(LexContext& context) const
{
	return std::make_unique<AsinhNode>(children[0]->copy(context));
}

// ACOSH NODE
AcoshNode::AcoshNode(std::unique_ptr<Node> child)
	: Node(child->context)
{
	children.emplace_back(std::move(child));
}

std::string AcoshNode::str() const
{
	return "acosh(" + children[0]->str() + ")";
}

std::string AcoshNode::glsl_str(const glsl::SymbolicContext& symtext) const
{
	return "acosh(" + children[0]->glsl_str(symtext) + ")";
}

std::unique_ptr<Node> AcoshNode::copy(LexContext& context) const
{
	return std::make_unique<AcoshNode>(children[0]->copy(context));
}

// ATANH NODE
AtanhNode::AtanhNode(std::unique_ptr<Node> child)
	: Node(child->context)
{
	children.emplace_back(std::move(child));
}

std::string AtanhNode::str() const
{
	return "atanh(" + children[0]->str() + ")";
}

std::string AtanhNode::glsl_str(const glsl::SymbolicContext& symtext) const
{
	return "atanh(" + children[0]->glsl_str(symtext) + ")";
}

std::unique_ptr<Node> AtanhNode::copy(LexContext& context) const
{
	return std::make_unique<AtanhNode>(children[0]->copy(context));
}

// DERIVATIVE NODE

expression::DerivativeNode::DerivativeNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child)
	: Node(left_child->context)
{
	const VariableNode* var_ptr = dynamic_cast<const VariableNode*>(right_child.get());
	if (var_ptr == nullptr) {
		throw std::runtime_error("Right argument in DerivativeNode must be a VariableNode");
	}

	auto abs_ptr = dynamic_cast<const AbsNode*>(left_child.get());
	if (abs_ptr != nullptr) {
		std::unique_ptr<Node> lc = std::move(left_child->children[0]);
		std::unique_ptr<Node> rc = lc->diff(right_child->str());

		std::unique_ptr<Node> schild = std::make_unique<SgnNode>(std::move(lc));

		std::unique_ptr<Node> child = std::make_unique<MulNode>(std::move(schild), std::move(rc));

		children.clear();
		children.emplace_back(std::move(child));
		return;
	}

	auto sgn_ptr = dynamic_cast<const SgnNode*>(left_child.get());
	if (sgn_ptr != nullptr) {
		children.clear();
		children.emplace_back(std::make_unique<TokenNode>(ZeroToken(), context));
		return;
	}

	// This is the derivative of something non special
	{
		auto child = left_child->diff(right_child->str());
		children.clear();
		children.emplace_back(std::move(child));
	}
}

std::string DerivativeNode::str() const
{
	return children[0]->str();
}

std::string DerivativeNode::glsl_str(const glsl::SymbolicContext& symtext) const
{
	return children[0]->glsl_str(symtext);
}

std::unique_ptr<Node> DerivativeNode::copy(LexContext& context) const
{
	return std::make_unique<DerivativeNode>(children[0]->copy(context), children[1]->copy(context));
}

// SUBS NODE

expression::SubsNode::SubsNode(std::vector<std::unique_ptr<Node>>&& childs)
	: Node(std::move(childs))
{
	//throw std::runtime_error("Not Implemented Yet!");
	std::unordered_map<std::string, expression::Expression> substitutions;

	for (int i = 1; i < children.size(); i += 2) {

	}

}

std::string expression::SubsNode::str() const {
	return children[0]->str();
}

std::string expression::SubsNode::glsl_str(const glsl::SymbolicContext& symtext) const {
	return children[0]->glsl_str(symtext);
}

std::unique_ptr<Node> SubsNode::copy(LexContext& context) const
{
	std::vector<std::unique_ptr<Node>> copied_children;
	copied_children.reserve(children.size());

	for (auto& child : children) {
		copied_children.push_back(std::move(child->copy(context)));
	}

	return std::make_unique<SubsNode>(std::move(copied_children));
}

// EXPRESSION

Expression expression_creator(const std::string& expression, const LexContext& context)
{
	std::string expr = util::to_lower_case(util::remove_whitespace(expression));

	Lexer lexer(context);

	auto toks = lexer.lex(expr);

	Shunter shunter;
	auto shunter_toks = shunter.shunt(std::move(toks));

	return Expression(context, shunter_toks, Expression::default_expression_creation_map());
}

Expression expression_creator(const std::string& expression, const std::vector<std::string>& variables)
{
	std::string expr = util::to_lower_case(util::remove_whitespace(expression));

	LexContext context;
	for (auto var : variables) {
		var = util::to_lower_case(var);
		context.variables.emplace_back(var);
		context.variable_assumptions.insert(SymEngine::contains(SymEngine::symbol(var), SymEngine::reals()));
	}

	Lexer lexer(context);

	auto toks = lexer.lex(expr);

	Shunter shunter;
	auto shunter_toks = shunter.shunt(std::move(toks));
	return Expression(context, shunter_toks, Expression::default_expression_creation_map());
}

Expression::Expression(const std::string& expression, const std::vector<std::string>& variables)
: Expression(expression_creator(expression, variables))
{}

Expression::Expression(const std::string& expression, const LexContext& context)
: Expression(expression_creator(expression, context))
{}

Expression::Expression(const std::unique_ptr<Node>& root_child, const LexContext& context)
	: m_Context(context), Node(m_Context)
{
	children.emplace_back(root_child->copy(m_Context));
	m_Expression = root_child->str();
}

Expression::Expression(const std::unique_ptr<Node>& root_child, const LexContext& context, const std::string& expr)
	: m_Context(context), Node(m_Context), m_Expression(expr)
{
	children.emplace_back(root_child->copy(m_Context));
}

Expression::Expression(const LexContext& context, const std::deque<std::unique_ptr<Token>>& tokens,
	const ExpressionCreationMap& creation_map)
	: m_Context(context), Node(m_Context)
{
	std::vector<std::unique_ptr<Node>> nodes;

	for (auto& token : tokens) {
		auto creation_func = creation_map.at(token->get_id());
		creation_func(m_Context, *token, nodes);
	}

	if (nodes.size() != 1)
		throw std::runtime_error("Expression construction failed, more than one node was left after creation_map usage");

	for (auto& var : m_Context.variables) {
		std::string vname = var.name;
		std::transform(vname.begin(), vname.end(), vname.begin(), [](unsigned char c) { return std::tolower(c); });
		m_Variables.emplace_back(vname);
	}

	children.emplace_back(std::move(nodes[0]));
}

std::string Expression::str() const
{
	return children[0]->str();
}

std::string Expression::glsl_str(const glsl::SymbolicContext& symtext) const
{
	return children[0]->glsl_str(symtext);
}

bool Expression::is_zero() const
{
	auto& child = children[0];
	TokenNode* child_node = dynamic_cast<TokenNode*>(child.get());
	if (child_node != nullptr) {
		ZeroToken* zero_node = dynamic_cast<ZeroToken*>(child_node->pToken.get());
		if (zero_node != nullptr) {
			return true;
		}
	}
	return false;
}

std::unique_ptr<Node> Expression::copy(LexContext& context) const
{
	return std::make_unique<Expression>(children[0], context, m_Expression);
}

ExpressionCreationMap Expression::default_expression_creation_map() {
	return ExpressionCreationMap{
		// Fixed Tokens
		{FixedIDs::UNITY_ID,
		[](LexContext& context, const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
			{
				nodes.push_back(node_from_token(tok, context));
			}
		},
		{FixedIDs::NEG_UNITY_ID,
		[](LexContext& context, const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
			{
				nodes.push_back(node_from_token(tok, context));
			}
		},
		{FixedIDs::ZERO_ID,
		[](LexContext& context, const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
			{
				nodes.push_back(node_from_token(tok, context));
			}
		},
		{FixedIDs::NAN_ID,
		[](LexContext& context, const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
			{
				nodes.push_back(node_from_token(tok, context));
			}
		},
		{FixedIDs::NUMBER_ID,
		[](LexContext& context, const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
			{
				nodes.push_back(node_from_token(tok, context));
			}
		},
		{FixedIDs::VARIABLE_ID,
		[](LexContext& context, const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
			{
				const VariableToken& vtok = static_cast<const VariableToken&>(tok);
				nodes.push_back(std::make_unique<VariableNode>(vtok, context));
			}
		},
		// Operators
		{DefaultOperatorIDs::NEG_ID,
		[](LexContext& context, const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
			{
				auto node = std::make_unique<NegNode>(std::move(nodes.back()));
				nodes.pop_back();
				nodes.push_back(std::move(node));
			}
		},
		{DefaultOperatorIDs::POW_ID,
		[](LexContext& context, const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
			{
				auto rc = std::move(nodes.back());
				nodes.pop_back();
				auto lc = std::move(nodes.back());
				nodes.pop_back();

				nodes.push_back(std::make_unique<PowNode>(std::move(lc), std::move(rc)));
			}
		},
		{DefaultOperatorIDs::MUL_ID,
		[](LexContext& context, const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
			{
				auto rc = std::move(nodes.back());
				nodes.pop_back();
				auto lc = std::move(nodes.back());
				nodes.pop_back();

				nodes.push_back(std::make_unique<MulNode>(std::move(lc), std::move(rc)));
			}
		},
		{DefaultOperatorIDs::DIV_ID,
		[](LexContext& context, const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
			{
				auto rc = std::move(nodes.back());
				nodes.pop_back();
				auto lc = std::move(nodes.back());
				nodes.pop_back();

				nodes.push_back(std::make_unique<DivNode>(std::move(lc), std::move(rc)));
			}
		},
		{DefaultOperatorIDs::ADD_ID,
		[](LexContext& context, const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
			{
				auto rc = std::move(nodes.back());
				nodes.pop_back();
				auto lc = std::move(nodes.back());
				nodes.pop_back();

				nodes.push_back(std::make_unique<AddNode>(std::move(lc), std::move(rc)));
			}
		},
		{DefaultOperatorIDs::SUB_ID,
		[](LexContext& context, const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
			{
				auto rc = std::move(nodes.back());
				nodes.pop_back();
				auto lc = std::move(nodes.back());
				nodes.pop_back();

				nodes.push_back(std::make_unique<SubNode>(std::move(lc), std::move(rc)));
			}
		},
		// Functions
		// Binary
		{ DefaultFunctionIDs::POW_ID,
		[](LexContext& context, const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
			{
				auto rc = std::move(nodes.back());
				nodes.pop_back();
				auto lc = std::move(nodes.back());
				nodes.pop_back();

				nodes.push_back(std::make_unique<PowNode>(std::move(lc), std::move(rc)));
			}
		},

		// Unary
		{ DefaultFunctionIDs::ABS_ID,
		[](LexContext& context, const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
			{
				auto node = std::make_unique<AbsNode>(std::move(nodes.back()));
				nodes.pop_back();
				nodes.push_back(std::move(node));
			}
		},
		{ DefaultFunctionIDs::SQRT_ID,
		[](LexContext& context, const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
			{
				auto node = std::make_unique<SqrtNode>(std::move(nodes.back()));
				nodes.pop_back();
				nodes.push_back(std::move(node));
			}
		},
		{ DefaultFunctionIDs::EXP_ID,
		[](LexContext& context, const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
			{
				auto node = std::make_unique<ExpNode>(std::move(nodes.back()));
				nodes.pop_back();
				nodes.push_back(std::move(node));
			}
		},
		{ DefaultFunctionIDs::LOG_ID,
		[](LexContext& context, const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
			{
				auto node = std::make_unique<LogNode>(std::move(nodes.back()));
				nodes.pop_back();
				nodes.push_back(std::move(node));
			}
		},
		// Trig
		{DefaultFunctionIDs::SIN_ID,
		[](LexContext& context, const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
			{
				auto node = std::make_unique<SinNode>(std::move(nodes.back()));
				nodes.pop_back();
				nodes.push_back(std::move(node));
			}
		},
		{ DefaultFunctionIDs::COS_ID,
		[](LexContext& context, const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
			{
				auto node = std::make_unique<CosNode>(std::move(nodes.back()));
				nodes.pop_back();
				nodes.push_back(std::move(node));
			}
		},
		{ DefaultFunctionIDs::TAN_ID,
		[](LexContext& context, const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
			{
				auto node = std::make_unique<TanNode>(std::move(nodes.back()));
				nodes.pop_back();
				nodes.push_back(std::move(node));
			}
		},
		{ DefaultFunctionIDs::ASIN_ID,
		[](LexContext& context, const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
			{
				auto node = std::make_unique<AsinNode>(std::move(nodes.back()));
				nodes.pop_back();
				nodes.push_back(std::move(node));
			}
		},
		{ DefaultFunctionIDs::ACOS_ID,
		[](LexContext& context, const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
			{
				auto node = std::make_unique<AcosNode>(std::move(nodes.back()));
				nodes.pop_back();
				nodes.push_back(std::move(node));
			}
		},
		{ DefaultFunctionIDs::ATAN_ID,
		[](LexContext& context, const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
			{
				auto node = std::make_unique<AtanNode>(std::move(nodes.back()));
				nodes.pop_back();
				nodes.push_back(std::move(node));
			}
		},
		{ DefaultFunctionIDs::SINH_ID,
		[](LexContext& context, const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
			{
				auto node = std::make_unique<SinhNode>(std::move(nodes.back()));
				nodes.pop_back();
				nodes.push_back(std::move(node));
			}
		},
		{ DefaultFunctionIDs::COSH_ID,
		[](LexContext& context, const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
			{
				auto node = std::make_unique<CoshNode>(std::move(nodes.back()));
				nodes.pop_back();
				nodes.push_back(std::move(node));
			}
		},
		{ DefaultFunctionIDs::TANH_ID,
		[](LexContext& context, const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
			{
				auto node = std::make_unique<TanhNode>(std::move(nodes.back()));
				nodes.pop_back();
				nodes.push_back(std::move(node));
			}
		},
		{ DefaultFunctionIDs::ASINH_ID,
		[](LexContext& context, const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
			{
				auto node = std::make_unique<AsinhNode>(std::move(nodes.back()));
				nodes.pop_back();
				nodes.push_back(std::move(node));
			}
		},
		{ DefaultFunctionIDs::ACOSH_ID,
		[](LexContext& context, const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
			{
				auto node = std::make_unique<AcoshNode>(std::move(nodes.back()));
				nodes.pop_back();
				nodes.push_back(std::move(node));
			}
		},
		{ DefaultFunctionIDs::ATANH_ID,
		[](LexContext& context, const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
			{
				auto node = std::make_unique<AtanhNode>(std::move(nodes.back()));
				nodes.pop_back();
				nodes.push_back(std::move(node));
			}
		},
		{ DefaultFunctionIDs::SGN_ID,
		[](LexContext& context, const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
			{
				auto node = std::make_unique<SgnNode>(std::move(nodes.back()));
				nodes.pop_back();
				nodes.push_back(std::move(node));
			}
		},
		{ DefaultFunctionIDs::DERIVATIVE_ID,
		[](LexContext& context, const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
			{
				auto rc = std::move(nodes.back());
				nodes.pop_back();
				auto lc = std::move(nodes.back());
				nodes.pop_back();

				nodes.push_back(std::make_unique<DerivativeNode>(std::move(lc), std::move(rc)));
			}
		},
		{ DefaultFunctionIDs::SUBS_ID,
		[](LexContext& context, const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
			{
				const FunctionToken* ftok = dynamic_cast<const FunctionToken*>(&tok);
				if (ftok == nullptr) {
					throw std::runtime_error("SUBS_ID creation map token was not a FunctionToken");
				}

				if (ftok->n_inputs % 2 != 1)
					throw std::runtime_error("SubsNode expects an odd number of arguments");

				std::vector<std::unique_ptr<Node>> children;
				children.reserve(ftok->n_inputs);
				for (int i = 0; i < ftok->n_inputs; ++i) {
					children.push_back(std::move(nodes.back()));
					nodes.pop_back();
				}

				nodes.push_back(std::make_unique<SubsNode>(std::move(children)));
			}
		}
	};
}

const LexContext& Expression::get_context() const
{
	return m_Context;
}

const std::string& Expression::get_expression() const
{
	return m_Expression;
}