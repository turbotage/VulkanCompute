module;

#include <memory>

export module expr;

import nodes;

namespace expression {

	export class Expr : public Node {
	public:

		Expr(std::unique_ptr<Node> root_child)
			: Node(root_child)
		{

		}


	};

}
