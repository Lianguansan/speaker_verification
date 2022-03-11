import glob
import os
 
# 拡張子.txtのファイルを取得する
for id in [1,2,3,4,5,6]:
    base_path = f"our_data/M00{id}test/"
    id_path = f"NM00{id}00"

    i = 0
    
    # txtファイルを取得する
    flist = glob.glob(base_path+"*.wav")
    
    # ファイル名を一括で変更する
    for file in flist:
        os.rename(file, base_path+id_path + str(i) + '_HS.wav')
        i+=1
 