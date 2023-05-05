#!/bin/bash
# this is a script that takes the atcoder dataset path as a parameter and moves it
# to the SLACC projects directory according to the same format as the other
# SLACC datasets

ATCODER_PATH=$(realpath $1)
SLACC_PROJECTS_PATH=$(realpath $2)
ATCODER_DATA_PATH=${ATCODER_PATH}/raw_data
SLACC_ATCODER_JAVA_PATH=${SLACC_PROJECTS_PATH}/src/main/java/atcoder
SLACC_ATCODER_PYTHON_PATH=${SLACC_PROJECTS_PATH}/src/main/python/atcoder

#INPUT --> ATCODER_PATH/raw_data/CX/PX/solutions.(py|java)
#OUTPUT -> SLACC_PROJECTS_PATH/src/main/java/atcoder/CX_PX/solution_name/sol.java
#       -> SLACC_PROJECTS_PATH/src/main/python/atcoder/CX_PX/solution_name/sol.py

#1) Create correct folder structure
     #java first

# TODO when ready to run the whole dataset remove the head -n 
echo "starting ..."
find ${ATCODER_PATH} -name "*.java" | head -n 4 | cut -d '/' -f 9,10,11 | sed 's/\//_/' | cut -d '.' -f 1 | sed 's/\//\/_/' | xargs -i mkdir -p ${SLACC_ATCODER_JAVA_PATH}/{}
# python second
find ${ATCODER_PATH} -name "*.py" | head -n 4 | cut -d '/' -f 9,10,11 | sed 's/\//_/' | cut -d '.' -f 1 | sed 's/\//\/_/' | xargs -i mkdir -p ${SLACC_ATCODER_PYTHON_PATH}/{}

echo "1/5 Created folder structure"
        # output --> CX_PX/_name

#2) Copy the files from one directory to the other
# TODO when ready to run the whole dataset remove the head -n 

#java first
find ${ATCODER_DATA_PATH} -name "*.java" | head -n 4 | rev | cut -d '/' -f 1,2,3 | rev | cut -d '.' -f 1 | awk -v p1=${ATCODER_DATA_PATH} -v p2=${SLACC_ATCODER_JAVA_PATH} -F / '{print "cp "p1"/"$1"/"$2"/"$3".java " p2"/"$1"_"$2"/_"$3 }' | bash

#python second
find ${ATCODER_DATA_PATH} -name "*.py" | head -n 4 | rev | cut -d '/' -f 1,2,3 | rev | cut -d '.' -f 1 | awk -v p1=${ATCODER_DATA_PATH} -v p2=${SLACC_ATCODER_PYTHON_PATH} -F / '{print "cp "p1"/"$1"/"$2"/"$3".py " p2"/"$1"_"$2"/_"$3 }' | bash

echo "2/5 Copied code into folders"

#rename the java files to Main.java
find ${SLACC_ATCODER_JAVA_PATH} -name "*.java" | xargs -I{} dirname {} | xargs -I{} bash -c "mv {}/*.java {}/Main.java"

#renaming the python files to main.py
find ${SLACC_ATCODER_PYTHON_PATH} -name "*.py" | xargs -I{} dirname {} | xargs -I{} bash -c "mv {}/*.py {}/main.py"

#indenting and adding _main methods
#sed -i 's/^/\t/g' test.py
find ${SLACC_ATCODER_PYTHON_PATH} -name 'main.py' | xargs -I{} sed -i 's/^/\t/g' {}
# sed -i -e '1s/^/def _main():\n/' test.py
find ${SLACC_ATCODER_PYTHON_PATH} -name 'main.py' | xargs -I{} sed -i -e '1s/^/def _main():\n/' {}

echo "3/5 Renaming code files"

#add a package declaration to the front of each file
find ${SLACC_ATCODER_JAVA_PATH} -name "Main.java" | awk -v q="'" -F / '{print "sed -i -e " q "1s;^;package atcoder."$(NF-2)"."$(NF-1)"\;\\n;" q " "$0}' | bash

echo "4/5 added package declarations"

# /projects/src/main/python/atcoder/CX_PY/DDDDDD.py
# touch __init__.py in the atcoder folder
touch ${SLACC_ATCODER_PYTHON_PATH}/__init__.py

# touch __init__ file in CX_PY folder
ls -l ${SLACC_ATCODER_PYTHON_PATH} | grep d | rev | cut -d " " -f 1 | rev | xargs -I{} touch ${SLACC_ATCODER_PYTHON_PATH}/{}/__init__.py

# add the input files for each atcoder problem
find ${ATCODER_DATA_PATH} -name "*.txt" | head -n 1 | rev | cut -d '/' -f 1,2,3 | rev | awk -v p1="${ATCODER_DATA_PATH}/" -v p2="${SLACC_ATCODER_PYTHON_PATH}/" -F / '{print "cp  " p1$0 " "p2$1"_"$2}'  | bash


# touch __init__ file in python folders so they can be mported 
# (DDDD.py folder)
find ${SLACC_ATCODER_PYTHON_PATH} -name "*.py" | xargs -I{} dirname {} | xargs -I{} touch {}/__init__.py


#echo "5/5 added package __init__.py files, input files and renaming"

echo "Completed"
