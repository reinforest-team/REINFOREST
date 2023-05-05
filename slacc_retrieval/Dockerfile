#this docker file sets up a container for SLACC to run in
# atm it just sets up the container and doesnt actually run anything



#0) base image ubuntu 20.04 atm
FROM ubuntu:focal as BASE

#takes command line argument to find the local installation of slacc fork
#ARG SLACC_LOCAL="../../reinforest/SLACC_fork"
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update

#General Dependencies
RUN apt-get -y install gnupg
RUN apt-get -y install apt-utils
RUN apt-get -y install systemctl
RUN apt-get install -y git
RUN apt-get -y install wget
RUN apt-get -y install curl
RUN apt-get -y install vim
RUN apt-get install -y --no-install-recommends \
      ca-certificates \
      jq \
      numactl
RUN apt-get install -y dirmngr gnupg apt-transport-https ca-certificates software-properties-common

#1)  Setup and run mongo db
#    NOTE: originally tried with Mongo 6 but doesn't work due to deprecated API calls
RUN wget -qO - https://www.mongodb.org/static/pgp/server-5.0.asc | apt-key add -
RUN touch /etc/apt/sources.list.d/mongodb-org-5.0.list
RUN echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/5.0 multiverse" | tee /etc/apt/sources.list.d/mongodb-org-5.0.list
RUN apt-get -y update
RUN apt-get install -y mongodb-org
EXPOSE 27017
#1a) Setup MONGO_HOME 
ENV MONGO_HOME "/usr/bin"
RUN echo localhost > /usr/bin/host_machine.txt
RUN export MONGO_HOME


#2) Java 1.8+
RUN apt-get install -y openjdk-8-jdk
RUN apt-get install -y ant
RUN apt-get clean;

#3) Maven 3.3+
RUN apt-get install -y maven

#4) Python 2.7.6+
RUN apt-get install -y python2-dev

#5) pip 9+
RUN curl https://bootstrap.pypa.io/pip/2.7/get-pip.py -o get-pip.py
RUN python2 get-pip.py
RUN ln -s /usr/bin/python2 /usr/bin/python

#6) clone slacc
#RUN git clone https://github.com/DynamicCodeSearch/SLACC.git /home/SLACC
#6b) adding modified SLACC from local directory
#   NOTE: mounting here because want to keep results accessible to host instead of limited to container
#COPY $SLACC_LOCAL /home/SLACC
#Add the SLACC_LOCAL directory with the docker run command this way its accessible from outside the container

#7) install R for slacc dependencies
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
RUN add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu focal-cran40/'
RUN apt-get install -y r-base

#8) install python slacc dependencies
RUN wget https://raw.githubusercontent.com/DynamicCodeSearch/SLACC/master/code/requirements.txt
RUN test -f requirements.txt
RUN pip install -r requirements.txt

#8) configure slacc files
# these commands are not right NEED EDITING
#RUN FILEPATH="/home/SLACC/code/src/main/java/edu/ncsu/config/Settings.java" && sed -i 's/Utils.pathJoin(HOME, "Raise", "ProgramRepair");/"\/home"/' $FILEPATH
#RUN FILEPATH="/home/SLACC/code/src/main/python/properties.py" && sed -i 's/os.path.join(HOME, "Raise", "ProgramRepair")/"\/home"/' $FILEPATH
# TODO: adjust python snipping script to have the correct path
# TODO remove .in and .out files from the code jam dataset

#9)Make mongodb db file
RUN mkdir -p /data/db
# mkdir -p /data/db && mongod --fork --logpath /var/log/mongodb/mongod.log
#Default commands when container is run

#10) Copy the startup script into the home directory
COPY start_script.sh /home

CMD ["mongod"]
