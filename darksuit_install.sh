#!/bin/bash

# Make script executable
chmod +x $0

# Install build tools and OpenCV
sudo apt-get install -y build-essential git libopencv-dev

# Clone and build Darknet
mkdir src
cd src
git clone https://github.com/AlexeyAB/darknet.git
cd darknet

GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader)
NEW_ARCH="ARCH= -gencode arch=compute_${GPU_ARCH//./},code=[sm_${GPU_ARCH//./},compute_${GPU_ARCH//./}]"
sed -i "0,/^ARCH=/s//${NEW_ARCH}\n#/" Makefile
sed -i '1,/^-gencode arch=compute_/! {/^-gencode arch=compute_/s/^/#/}' Makefile
sed -i 's/OPENCV=0/OPENCV=1/' Makefile
sed -i 's/AVX=0/AVX=1/' Makefile
sed -i 's/OPENMP=0/OPENMP=1/' Makefile
sed -i 's/LIBSO=0/LIBSO=1/' Makefile
@echo saving...wait 2 seconds 
sleep 2 # Wait for 2 seconds to ensure file is saved
make 
sudo cp libdarknet.so /usr/local/lib/
sudo cp include/darknet.h /usr/local/include/
sudo ldconfig

# Install DarkHelp
cd ~/src
sudo apt-get install -y cmake build-essential libtclap-dev libmagic-dev libopencv-dev
git clone https://github.com/stephanecharette/DarkHelp.git
cd DarkHelp
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make j16
make package
sudo dpkg -i darkhelp*.deb

# Install DarkMark
cd ~/src
sudo apt-get install -y cmake libopencv-dev libx11-dev libfreetype6-dev libxrandr-dev libxinerama-dev libxcursor-dev libmagic-dev libpoppler-cpp-dev
git clone https://github.com/stephanecharette/DarkMark.git
cd DarkMark
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make j16
make package
sudo dpkg -i darkmark*.deb

