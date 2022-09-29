# SSRW 2022

Etudeの`~/users/ssrw2022`を個人のPCの好きなところにコピーする

`data/ROHAN4600` : dataset

`util/`: 画像 -> 動画, 動画 -> pt に変換するスクリプト

```
conda env create -f environment.yml

git branch -b <branch name>
git add .
git commit -m "commit"
git push
```

```
cd ssrw2022
mkdir build # do once
python train.py
```
