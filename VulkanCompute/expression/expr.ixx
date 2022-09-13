module;

#include <memory>
#include <deque>
#include <unordered_map>
#include <functional>

export module expr;

import token;
import nodes;

namespace expression {

	using ExpressionCreationMap = std::unordered_map<int32_t,
		std::function<void(const Token&, std::vector<std::unique_ptr<Node>>&)>>;

	export class Expression : public Node {
	public:

		Expression(std::unique_ptr<Node> root_child);

		Expression(const LexContext& context, const std::deque<std::unique_ptr<Token>>& tokens,
			ExpressionCreationMap& creation_map)
			: Node(context)
		{

		}

	};

}

