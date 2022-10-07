# SSRW 2022

Etudeの`~/users/ssrw2022`を個人のPCの好きなところにコピーする

`data/ROHAN4600` : dataset

`util/`: 画像 -> 動画, 動画 -> pt に変換するスクリプト

```
conda env create -f environment.yml
git pull origin main

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

data
```
1 ~ 100 zundamon normal
101 ~ 200 zundamon ama
201 ~ 300 zundamon tsun
301 ~ 400 zundamon sexy
401 ~ 724 zundamon recitation
725 ~ 824 itako normal
825 ~ 924 itako ama
925 ~ 1024 itako tsun
1025 ~ 1124 itako sexy
1125 ~ 1448 itako recitation
1449 ~ 1548 metan normal
1549 ~ 1648 metan ama
1649 ~ 1748 metan tsun
1749 ~ 1848 metan sexy
1849 ~ 2172 metan recitation
2173 ~ 2272 sora normal
2273 ~ 2372 sora ama
2373 ~ 2472 sora tsun
2473 ~ 2572 sora sexy
2573 ~ 2672 sora whis
2673 ~ 2996 sora recitation
2997 ~ 7596 ROHAN4600
```
