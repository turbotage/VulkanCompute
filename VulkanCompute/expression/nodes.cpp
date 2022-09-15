module;

#include <string>
#include <set>
#include <memory>
#include <vector>

#include <symengine/expression.h>
#include <symengine/simplify.h>
#include <symengine/parser.h>

module nodes;

import expr;
import util;

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

std::unique_ptr<expression::Node> expression::Node::diff(const std::string& x)
{
	std::string expr_str = util::to_lower_case(util::remove_whitespace(str()));

	SymEngine::Assumptions assum1(context.variable_assumptions);

	auto parsed = SymEngine::simplify(SymEngine::parse(expr_str), &assum1);
	auto dexpr = parsed->diff(SymEngine::symbol(
		util::to_lower_case(
			util::remove_whitespace(x))));
	
	std::set<std::string> vars;
	symengine_get_args(dexpr, vars);

	context.variables.reserve(vars.size());
	for (auto& var : vars) {
		if (std::find(context.variables.begin(), context.variables.end(), var) == context.variables.end()) {
			context.variables.emplace_back(var, context);
		}
		context.variable_assumptions.insert(SymEngine::contains(SymEngine::symbol(var), SymEngine::reals()));
	}
	
	SymEngine::Assumptions assum2(context.variable_assumptions);

	auto sim_dexpr = SymEngine::simplify(dexpr, &assum2);

	auto dexpr_str = util::to_lower_case(
			util::remove_whitespace(sim_dexpr->__str__()));

	auto ret = std::make_unique<expression::Expression>(dexpr_str, context);
	return ret;
}

expression::DerivativeNode::DerivativeNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child)
	: Node(left_child->context)
{
	auto var_ptr = dynamic_cast<const VariableNode*>(right_child.get());
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
}

std::string expression::SubsNode::str() {
	return "";
}

std::string expression::SubsNode::glsl_str() {
	return "";
}
