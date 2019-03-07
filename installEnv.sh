#!/bin/sh
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-5.3.1-MacOSX-x86_64.sh 
chmod +x Anaconda3-5.3.1-MacOSX-x86_64.sh
sh Anaconda3-5.3.1-MacOSX-x86_64.sh  
conda config –add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ 
conda config –add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ 
conda config –add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/ 
conda config –set show_channel_urls yes 
vim ~/.condarc 
delete “- defaults” line 
conda create –name tf python=3.6
source activate tf 
pip install -r requirements.txt