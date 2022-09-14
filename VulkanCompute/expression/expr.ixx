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

namespace expression {

	using ExpressionCreationMap = std::unordered_map<int32_t,
		std::function<void(const Token&, std::vector<std::unique_ptr<Node>>&)>>;

	export class Expression;
	export Expression expression_creator(const std::string& expression, std::vector<std::string> variables);

	export class Expression : public Node {
	public:

		Expression(Expression&&) = default;

		Expression(const std::string& expression, std::vector<std::string> variables)
			: Expression(expression_creator(expression, variables)) 
		{}

		Expression(std::unique_ptr<Node> root_child)
			: Node(root_child->context)
		{
			children.emplace_back(std::move(root_child));
		}

		Expression(const LexContext& context, const std::deque<std::unique_ptr<Token>>& tokens,
			const ExpressionCreationMap& creation_map)
			: Node(context), m_Context(context)
		{
			std::vector<std::unique_ptr<Node>> nodes;

			for (auto& token : tokens) {
				auto creation_func = creation_map.at(token->get_id());
				creation_func(*token, nodes);
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

		std::string str() override {
			return children[0]->str();
		}

		std::string glsl_str() override {
			return children[0]->glsl_str();
		}

		/*
		std::string diff_str(const std::string& x) {
			return diff(x)->str();
		}

		std::string diff_glsl_str(const std::string& x) {
			return diff(x)->glsl_str();
		}
		*/

		/*
		Expression diff_expr(const std::string& x) {
			std::string expr_str = str();
			auto parsed = SymEngine::parse(expr_str);
			auto xsym = SymEngine::symbol(util::to_lower_case(x));
			auto diffstr = parsed->diff(xsym);
			expr_str = diffstr->__str__();
			expr_str = util::to_lower_case(util::remove_whitespace(expr_str));
			return Expression(expr_str, m_Variables);
		}
		*/

		static ExpressionCreationMap default_expression_creation_map(const LexContext& context) {
			return ExpressionCreationMap{
				// Fixed Tokens
				{FixedIDs::UNITY_ID,
				[&context](const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
					{
						nodes.push_back(node_from_token(tok, context));
					}
				},
				{FixedIDs::NEG_UNITY_ID,
				[&context](const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
					{
						nodes.push_back(node_from_token(tok, context));
					}
				},
				{FixedIDs::ZERO_ID,
				[&context](const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
					{
						nodes.push_back(node_from_token(tok, context));
					}
				},
				{FixedIDs::NAN_ID,
				[&context](const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
					{
						nodes.push_back(node_from_token(tok, context));
					}
				},
				{FixedIDs::NUMBER_ID,
				[&context](const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
					{
						nodes.push_back(node_from_token(tok, context));
					}
				},
				{FixedIDs::VARIABLE_ID,
				[&context](const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
					{
						const VariableToken& vtok = static_cast<const VariableToken&>(tok);
						nodes.push_back(std::make_unique<VariableNode>(vtok, context));
					}
				},
				// Operators
				{DefaultOperatorIDs::NEG_ID,
				[&context](const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
					{
						auto node = std::make_unique<NegNode>(std::move(nodes.back()));
						nodes.pop_back();
						nodes.push_back(std::move(node));
					}
				},
				{DefaultOperatorIDs::POW_ID,
				[&context](const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
					{
						auto rc = std::move(nodes.back());
						nodes.pop_back();
						auto lc = std::move(nodes.back());
						nodes.pop_back();

						nodes.push_back(std::make_unique<PowNode>(std::move(lc), std::move(rc)));
					}
				},
				{DefaultOperatorIDs::MUL_ID,
				[&context](const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
					{
						auto rc = std::move(nodes.back());
						nodes.pop_back();
						auto lc = std::move(nodes.back());
						nodes.pop_back();

						nodes.push_back(std::make_unique<MulNode>(std::move(lc), std::move(rc)));
					}
				},
				{DefaultOperatorIDs::DIV_ID,
				[&context](const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
					{
						auto rc = std::move(nodes.back());
						nodes.pop_back();
						auto lc = std::move(nodes.back());
						nodes.pop_back();

						nodes.push_back(std::make_unique<DivNode>(std::move(lc), std::move(rc)));
					}
				},
				{DefaultOperatorIDs::ADD_ID,
				[&context](const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
					{
						auto rc = std::move(nodes.back());
						nodes.pop_back();
						auto lc = std::move(nodes.back());
						nodes.pop_back();

						nodes.push_back(std::make_unique<AddNode>(std::move(lc), std::move(rc)));
					}
				},
				{DefaultOperatorIDs::SUB_ID,
				[&context](const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
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
				[&context](const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
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
				[&context](const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
					{
						auto node = std::make_unique<AbsNode>(std::move(nodes.back()));
						nodes.pop_back();
						nodes.push_back(std::move(node));
					}
				},
				{ DefaultFunctionIDs::SQRT_ID,
				[&context](const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
					{
						auto node = std::make_unique<SqrtNode>(std::move(nodes.back()));
						nodes.pop_back();
						nodes.push_back(std::move(node));
					}
				},
				{ DefaultFunctionIDs::EXP_ID,
				[&context](const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
					{
						auto node = std::make_unique<ExpNode>(std::move(nodes.back()));
						nodes.pop_back();
						nodes.push_back(std::move(node));
					}
				},
				{ DefaultFunctionIDs::LOG_ID,
				[&context](const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
					{
						auto node = std::make_unique<LogNode>(std::move(nodes.back()));
						nodes.pop_back();
						nodes.push_back(std::move(node));
					}
				},
				// Trig
				{DefaultFunctionIDs::SIN_ID,
				[&context](const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
					{
						auto node = std::make_unique<SinNode>(std::move(nodes.back()));
						nodes.pop_back();
						nodes.push_back(std::move(node));
					}
				},
				{ DefaultFunctionIDs::COS_ID,
				[&context](const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
					{
						auto node = std::make_unique<CosNode>(std::move(nodes.back()));
						nodes.pop_back();
						nodes.push_back(std::move(node));
					}
				},
				{ DefaultFunctionIDs::TAN_ID,
				[&context](const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
					{
						auto node = std::make_unique<TanNode>(std::move(nodes.back()));
						nodes.pop_back();
						nodes.push_back(std::move(node));
					}
				},
				{ DefaultFunctionIDs::ASIN_ID,
				[&context](const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
					{
						auto node = std::make_unique<AsinNode>(std::move(nodes.back()));
						nodes.pop_back();
						nodes.push_back(std::move(node));
					}
				},
				{ DefaultFunctionIDs::ACOS_ID,
				[&context](const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
					{
						auto node = std::make_unique<AcosNode>(std::move(nodes.back()));
						nodes.pop_back();
						nodes.push_back(std::move(node));
					}
				},
				{ DefaultFunctionIDs::ATAN_ID,
				[&context](const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
					{
						auto node = std::make_unique<AtanNode>(std::move(nodes.back()));
						nodes.pop_back();
						nodes.push_back(std::move(node));
					}
				},
				{ DefaultFunctionIDs::SINH_ID,
				[&context](const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
					{
						auto node = std::make_unique<SinhNode>(std::move(nodes.back()));
						nodes.pop_back();
						nodes.push_back(std::move(node));
					}
				},
				{ DefaultFunctionIDs::COSH_ID,
				[&context](const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
					{
						auto node = std::make_unique<CoshNode>(std::move(nodes.back()));
						nodes.pop_back();
						nodes.push_back(std::move(node));
					}
				},
				{ DefaultFunctionIDs::TANH_ID,
				[&context](const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
					{
						auto node = std::make_unique<TanhNode>(std::move(nodes.back()));
						nodes.pop_back();
						nodes.push_back(std::move(node));
					}
				},
				{ DefaultFunctionIDs::ASINH_ID,
				[&context](const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
					{
						auto node = std::make_unique<AsinhNode>(std::move(nodes.back()));
						nodes.pop_back();
						nodes.push_back(std::move(node));
					}
				},
				{ DefaultFunctionIDs::ACOSH_ID,
				[&context](const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
					{
						auto node = std::make_unique<AcoshNode>(std::move(nodes.back()));
						nodes.pop_back();
						nodes.push_back(std::move(node));
					}
				},
				{ DefaultFunctionIDs::ATANH_ID,
				[&context](const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
					{
						auto node = std::make_unique<AtanhNode>(std::move(nodes.back()));
						nodes.pop_back();
						nodes.push_back(std::move(node));
					}
				},
				{ DefaultFunctionIDs::DERIVATIVE_ID,
				[&context](const Token& tok, std::vector<std::unique_ptr<Node>>& nodes)
					{
						auto rc = std::move(nodes.back());
						nodes.pop_back();
						auto lc = std::move(nodes.back());
						nodes.pop_back();

						nodes.push_back(std::make_unique<DerivativeNode>(std::move(lc), std::move(rc)));
					}
				},

			};
		}
		
	private:

	private:

		LexContext m_Context;
		std::string m_Expression;
		std::vector<std::string> m_Variables;
	};

	export Expression expression_creator(const std::string& expression, std::vector<std::string> variables)
	{
		std::string expr = util::to_lower_case(util::remove_whitespace(expression));

		LexContext context;
		for (auto& var : variables) {
			std::transform(var.begin(), var.end(), var.begin(), [](unsigned char c) { return std::tolower(c); });
			context.variables.emplace_back(var);
		}

		Lexer lexer(std::move(context));

		auto toks = lexer.lex(expr);

		Shunter shunter;
		auto shunter_toks = shunter.shunt(std::move(toks));
		return Expression(context, shunter_toks, Expression::default_expression_creation_map(context));
	}

}

