#!/bin/bash

# Check if installed softwares

function program_is_installed {
  local return_=1
  type $1 >/dev/null 2>&1 || { local return_=0; }
  echo "$return_"
}

if [ $(program_is_installed git) = 0 ]
then
    echo "Git is not installed"
    echo "Please install git first to use this code"
    echo "For example: sudo apt install -y git"
    exit 1
fi

# Initialize darknet
git clone https://github.com/pjreddie/darknet.git
cd darknet
make -j8

wget https://pjreddie.com/media/files/yolov3.weights

# Initialize project
cd ..
mkdir input
mkdir output

echo "Done initializing project"