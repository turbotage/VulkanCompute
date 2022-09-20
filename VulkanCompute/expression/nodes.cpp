module;

#include <string>
#include <set>
#include <memory>
#include <vector>
#include <complex>

#include <symengine/expression.h>
#include <symengine/simplify.h>
#include <symengine/parser.h>

module nodes;

import expr;
import util;
import lexer;

void expression::Node::fill_variable_list(std::set<std::string>& vars)
{
	VariableNode* var_node = dynamic_cast<VariableNode*>(this);
	if (var_node != nullptr) {
		vars.insert(var_node->str());
	}

	for (auto& child : children) {
		child->fill_variable_list(vars);
	}
}

std::unique_ptr<expression::Expression> expression::Node::diff(const std::string& x) const
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
