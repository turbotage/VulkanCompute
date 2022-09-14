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
	std::set<std::string> vars;
	fill_variable_list(vars);
	
	std::vector<std::string> vector_vars(vars.begin(), vars.end());

	auto parsed = SymEngine::parse(expr_str);
	auto dexpr = parsed->diff(SymEngine::symbol(
		util::to_lower_case(
			util::remove_whitespace(x))))->__str__();

	return std::make_unique<expression::Expression>(dexpr, vector_vars);
}

expression::DerivativeNode::DerivativeNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child)
	: Node(left_child->context)
{
	auto child_ptr = dynamic_cast<const AbsNode*>(left_child.get());
	if (child_ptr != nullptr) {
		std::unique_ptr<Node> lc = std::move(left_child->children[0]);
		std::unique_ptr<Node> rc = lc->diff(right_child->str());

		std::unique_ptr<Node> schild = std::make_unique<SgnNode>(std::move(lc));

		std::unique_ptr<Node> child = std::make_unique<MulNode>(std::move(schild), std::move(rc));

		children.clear();
		children.emplace_back(std::move(child));
	}
	else
	{
		auto child = left_child->diff(right_child->str());
		children.clear();
		children.emplace_back(std::move(child));
	}

}
