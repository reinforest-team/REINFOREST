import json
from typing import Any, Dict, Optional
from tqdm import tqdm
from tree_sitter import Language, Parser, Node
import os
import sys
from apted import APTED, Config
import traceback

if True:
    project_dir = os.path.abspath(
        os.path.join(os.path.abspath(__file__), "../../.."))
    if project_dir not in sys.path:
        sys.path.append(project_dir)
    parser_path = os.path.join(project_dir, 'parser', 'languages.so')
    from_java = os.path.join(
        project_dir, 'parser', 'node_configs', 'from_java.json'
    )
    from_python = os.path.join(
        project_dir, 'parser', 'node_configs', 'from_python.json'
    )
    PY_LANGUAGE = Language(parser_path, 'python')
    JAVA_LANGUAGE = Language(parser_path, 'java')
    from_java_config = json.load(open(from_java, 'r'))
    from_python_config = json.load(open(from_python, 'r'))
    from source import util
    logger = util.get_logger()


class TreeDisctanceConfig(Config):
    def rename(
        self,
        node1: Dict[str, Any],
        node2: Dict[str, Any]
    ):
        return 1 if node1['type'] != node2['type'] else 0

    def children(
        self, node: Dict[str, Any]
    ):
        try:
            return node['children']
        except:
            print(node)


def map_tree(
    root: Node,
    config: Dict[str, str]
):
    node = {'type': root.type, 'children': []}
    if node['type'] in config:
        node['type'] = config[node['type']]
    else:
        node['type'] = "unk"
    for child in root.children:
        if child is not None:
            node['children'].append(
                map_tree(child, config)
            )
    return node


def compute_distance(tree1, tree2):
    apted = APTED(tree1, tree2, TreeDisctanceConfig())
    return apted.compute_edit_distance()


class ASTSearch:
    def __init__(
        self,
        source_lang: str,
        target_lang: str,
    ):
        global PY_LANGUAGE, JAVA_LANGUAGE
        self.source_lang = source_lang
        self.target_lang = target_lang
        if self.source_lang == 'python':
            self.source_language = PY_LANGUAGE
            self.target_language = JAVA_LANGUAGE
            self.source_config = from_python_config
            self.target_config = from_java_config
        else:
            self.source_language = JAVA_LANGUAGE
            self.target_language = PY_LANGUAGE
            self.source_config = from_java_config
            self.target_config = from_python_config
        self.apted_config = TreeDisctanceConfig()

    def calculate_score(self, query_tree, corpus_tree):
        apted = APTED(query_tree, corpus_tree, self.apted_config)
        ed = apted.compute_edit_distance()
        return 1 / (1 + ed)

    def get_scores(self, code, corpus):
        src_parser = Parser()
        src_parser.set_language(self.source_language)
        query_tree = src_parser.parse(bytes(code, "utf8")).root_node
        tgt_parser = Parser()
        tgt_parser.set_language(self.target_language)
        corpus_trees = [
            tgt_parser.parse(bytes(c, "utf8")).root_node for c in corpus
        ]
        query_tree = map_tree(query_tree, self.source_config)
        corpus_trees = [
            map_tree(
                corpus_tree, self.target_config
            ) for corpus_tree in corpus_trees
        ]
        scores = [
            self.calculate_score(
                query_tree, corpus_tree
            ) for corpus_tree in corpus_trees
        ]
        return scores


def count_nodes(node: Dict[str, Any]):
    return 1 + sum(count_nodes(child) for child in node['children'])


if __name__ == "__main__":
    java_code = """public class Test {
        public static void main(String[] args) {
            System.out.println("Hello World!");
        }
    }"""

    python_code = """def get_scores(self, code, corpus):
        if self.tree_cache is None:
            parser = Parser()
            parser.set_language(self.source_language)
            query_tree = parser.parse(bytes(code, "utf8")).root_node
            corpus_trees = [
                parser.parse(bytes(c, "utf8")).root_node for c in corpus
            ]
            query_tree = map_tree(query_tree, self.source_config)
            corpus_trees = [
                map_tree(
                    corpus_tree, self.target_config
                ) for corpus_tree in corpus_trees
            ]
            scores = [
                self.calculate_score(
                    query_tree, corpus_tree
                ) for corpus_tree in corpus_trees
            ]
        else:
            if code not in self.tree_cache['code_to_keys']:
                return None
            query_idx = self.tree_cache['code_to_keys'][code]
            document_ids = [self.tree_cache['code_to_keys'][d] for d in corpus]
            if query_idx not in self.tree_cache['distances']:
                return None
            scores = [
                1 / (1 + self.tree_cache['distances'][query_idx][did])
                for did in document_ids
            ]
        return scores"""

    parser = Parser()
    parser.set_language(PY_LANGUAGE)
    query_tree = parser.parse(bytes(python_code, "utf8")).root_node
    original_map = map_tree(query_tree, from_python_config)
    # print(json.dumps(original_map, indent=4))
    print(count_nodes(original_map))
