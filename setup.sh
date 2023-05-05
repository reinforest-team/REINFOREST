#!/bin/bash

function setup_repo() {
    mkdir -p sitter-libs;
    git clone https://github.com/tree-sitter/tree-sitter-python sitter-libs/py;
    git clone https://github.com/tree-sitter/tree-sitter-java sitter-libs/java;
    mkdir -p "parser";
    python setup_repo.py sitter-libs;
}

function create_and_activate() {
    conda create --name reinforest python=3.9;
    conda activate reinforest;
}

function install_deps() {
    pip install openai;
    pip install tiktoken;
    pip install nltk==3.8.1;
    conda install pytorch==1.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia;
    pip install transformers==4.16.2;
    pip install datasets==1.18.3;
    pip install scikit-learn==1.2.1;
}

#create_and_activate;
install_deps;
setup_repo;
