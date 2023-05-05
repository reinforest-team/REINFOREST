import antlr4
from antlr4.tree.Trees import Trees
from collections import Counter

from Python3Lexer import Python3Lexer
from Python3Parser import Python3Parser
from JavaLexer import JavaLexer
from JavaParser import JavaParser

# The compare_nodes and sim_score_trees functions remain the same as before

def get_python_ast(code):
    input_stream = antlr4.InputStream(code)
    lexer = Python3Lexer(input_stream)
    token_stream = antlr4.CommonTokenStream(lexer)
    parser = Python3Parser(token_stream)
    tree = parser.file_input()
    return tree

def get_java_ast(code):
    input_stream = antlr4.InputStream(code)
    lexer = JavaLexer(input_stream)
    token_stream = antlr4.CommonTokenStream(lexer)
    parser = JavaParser(token_stream)
    tree = parser.compilationUnit()
    return tree

def count_nodes_antlr(node):
    nodes = Counter({node.__class__.__name__: 1})
    for child in Trees.getChildren(node):
        nodes.update(count_nodes_antlr(child))
    return nodes

def sim_score_trees_antlr(tree1, tree2):
    node_count1 = count_nodes_antlr(tree1)
    node_count2 = count_nodes_antlr(tree2)

    total_nodes1 = sum(node_count1.values())
    total_nodes2 = sum(node_count2.values())
    common_nodes = sum((node_count1 & node_count2).values())

    sim_score = common_nodes / (total_nodes1 + total_nodes2 - common_nodes)
    return sim_score

# Example usage
python_code = '''
def foo(x):
    return x + 1
'''

java_code = '''
class Main {
    public static int foo(int x) {
        return x + 1;
    }
}
'''

python_tree = get_python_ast(python_code)
java_tree = get_java_ast(java_code)

print("Similarity score:", sim_score_trees_antlr(python_tree, java_tree))
