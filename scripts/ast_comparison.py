import ast
from collections import Counter

def compare_nodes(node1, node2):
    if type(node1) != type(node2):
        return False
    for field in node1._fields:
        field1 = getattr(node1, field)
        field2 = getattr(node2, field)
        if isinstance(field1, list) and isinstance(field2, list):
            if len(field1) != len(field2):
                return False
            for item1, item2 in zip(field1, field2):
                if not compare_nodes(item1, item2):
                    return False
        elif not (isinstance(field1, ast.AST) or isinstance(field2, ast.AST)):
            if field1 != field2:
                return False
    return True

def count_nodes(node):
    nodes = Counter({type(node).__name__: 1})
    for field in node._fields:
        field_value = getattr(node, field)
        if isinstance(field_value, list):
            for item in field_value:
                if isinstance(item, ast.AST):
                    nodes.update(count_nodes(item))
        elif isinstance(field_value, ast.AST):
            nodes.update(count_nodes(field_value))
    return nodes

def sim_score_trees(tree1, tree2):
    node_count1 = count_nodes(tree1)
    node_count2 = count_nodes(tree2)

    total_nodes1 = sum(node_count1.values())
    total_nodes2 = sum(node_count2.values())
    common_nodes = sum((node_count1 & node_count2).values())

    sim_score = common_nodes / (total_nodes1 + total_nodes2 - common_nodes)
    return sim_score

# Example usage
code1 = '''
def foo(x):
    return x + 1
'''

code2 = '''
def bar(y):
    return y + 1
'''

tree1 = ast.parse(code1)
tree2 = ast.parse(code2)

print("Similarity score:", sim_score_trees(tree1, tree2))

