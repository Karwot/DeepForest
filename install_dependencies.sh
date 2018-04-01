#! /usr/bin/env sh
cd ~
git clone https://github.com/LASzip/LASzip.git
cd LASzip
git checkout tags/2.0.2

mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=~/LASzip/build 
make
sudo make install

cd ~
unzip /home/travis/build/weecology/DeepForest/tests/LAStools
cd LAStools
make
sudo make install

