module;

#include <memory>
#include <deque>
#include <unordered_map>
#include <initializer_list>
#include <functional>
#include <stdexcept>
#include <vector>
#include <tuple>
#include <algorithm>
#include <set>
#include <iterator>
#include <optional>

#include <symengine/expression.h>
#include <symengine/simplify.h>
#include <symengine/parser.h>

export module expr;

import token;
import nodes;
import lexer;
import defaultexp;
import shunter;
import util;
import glsl;

namespace expression {

	using ExpressionCreationMap = std::unordered_map<int32_t,
		std::function<void(LexContext&, const Token&, std::vector<std::unique_ptr<Node>>&)>>;

	export void symengine_get_args(const SymEngine::RCP<const SymEngine::Basic>& subexpr, std::set<std::string>& args) 
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

	export std::string symengine_parse(const std::string& to_parse)
	{
		auto p = SymEngine::parse(to_parse);
		return p->__str__();
	}

	export class Expression;
	export Expression expression_creator(const std::string& expression, const LexContext& context);
	export Expression expression_creator(const std::string& expression, const std::vector<std::string>& variables);

	class Expression : public Node {
	public:

		Expression(Expression&&) = default;

		Expression(const std::string& expression, const std::vector<std::string>& variables)
			: Expression(expression_creator(expression, variables)) 
		{}

		Expression(const std::string& expression, const LexContext& context)
			: Expression(expression_creator(expression, context))
		{}

		Expression(const std::unique_ptr<Node>& root_child, const LexContext& context)
			: m_Context(context), Node(m_Context)
		{
			children.emplace_back(root_child->copy(m_Context));
			m_Expression = root_child->str();
		}

		Expression(const std::unique_ptr<Node>& root_child, const LexContext& context, const std::string& expr)
			: m_Context(context), Node(m_Context), m_Expression(expr)
		{
			children.emplace_back(root_child->copy(m_Context));
		}

		Expression(const LexContext& context, const std::deque<std::unique_ptr<Token>>& tokens,
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

		std::string str() const override 
		{
			return children[0]->str();
		}

		std::string glsl_str(const glsl::SymbolicContext& symtext) const override 
		{
			return children[0]->glsl_str(symtext);
		}

		std::unique_ptr<Node> copy(LexContext& context) const override
		{
			return std::make_unique<Expression>(children[0], context, m_Expression);
		}

		static ExpressionCreationMap default_expression_creation_map() {
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

		const LexContext& get_context() const
		{
			return m_Context;
		}

	private:

		LexContext m_Context;
		std::string m_Expression;
		std::vector<std::string> m_Variables;
	};

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

}

