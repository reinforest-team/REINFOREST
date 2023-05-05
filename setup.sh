#!/bin/bash

function setup_repo() {
    mkdir -p sitter-libs;
    git clone https://github.com/tree-sitter/tree-sitter-go sitter-libs/go;
    git clone https://github.com/tree-sitter/tree-sitter-javascript sitter-libs/js;
    git clone https://github.com/tree-sitter/tree-sitter-c sitter-libs/c;
    git clone https://github.com/tree-sitter/tree-sitter-cpp sitter-libs/cpp;
    git clone https://github.com/tree-sitter/tree-sitter-c-sharp sitter-libs/cs;
    git clone https://github.com/tree-sitter/tree-sitter-python sitter-libs/py;
    git clone https://github.com/tree-sitter/tree-sitter-java sitter-libs/java;
    git clone https://github.com/tree-sitter/tree-sitter-ruby sitter-libs/ruby;
    git clone https://github.com/tree-sitter/tree-sitter-php sitter-libs/php;
    mkdir -p "parser";
    python setup_repo.py sitter-libs;
}

function create_and_activate() {
    conda create --name reinforest python=3.6;
    conda activate reinforest;
}

function install_deps() {
    conda install pytorch=1.6 torchvision torchaudio cudatoolkit=10.1 -c pytorch;
    pip install transformers==3.0.2;
    pip install tokenizers==0.10.3;
    pip install tree-sitter==0.19.0;
    # Please add the command if you add any package.
}

#create_and_activate;
install_deps;
setup_repo;