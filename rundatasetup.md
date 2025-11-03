cd CAPSTONE_L2R_ESS
mkdir data
cd data

wget https://ml-inat-competition-datasets.s3.amazonaws.com/2018/train_val2018.tar.gz
wget https://ml-inat-competition-datasets.s3.amazonaws.com/2018/train2018.json.tar.gz
wget https://ml-inat-competition-datasets.s3.amazonaws.com/2018/val2018.json.tar.gz

tar -xvzf train_val2018.tar.gz
tar -xvzf train2018.json.tar.gz
tar -xvzf val2018.json.tar.gz

rm train_val2018.tar.gz train2018.json.tar.gz val2018.json.tar.gz
