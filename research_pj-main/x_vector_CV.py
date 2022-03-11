# Pythonで長い会議を見える化〜話者ダイアリゼーションの動向〜
# https://qiita.com/toast-uz/items/44c6a12dbf10cb3055ca
# を拡張
#outputの保存先　/Output/
output_path = "output"
#modelの保存先　/model/
model_save_path =  "speech_model"
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torchaudio
from torchvision import transforms
import os
import numpy as np
import csv
from sklearn import metrics
import pickle

ENR_DATA_NUM = 6  # 登録人数。
TEST_DATA_NUM = 6 # テスト人数。
EACH_DATA_NUM = 9 # それぞれの人の登録したデータの数。
ROOT_PATH = 'our_data16_v2/'   ###### ここは登録ユーザの部分なのでとりあえずペンディングでok。
EPOCH_NUM =150
graph_save_name = f"{ROOT_PATH[:-1]}_{EPOCH_NUM}.png"
speakers = ['M001', 'M002', 'M003', 'M004', 'M005', 'M006', 'M007', 'M008', 'M009']  ######
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# soxはインストールが必要かも
torchaudio.set_audio_backend('soundfile') #windows だと torchaudio.set_audio_backend('soundfile') で動く？

class TrainDataset(Dataset):
    sample_rate = 16000
    def __init__(self, train=True, transform=None):
        self.transform = transform

        split_rate=0.8
        original_data_path = './CommonVoice/cv-corpus-5.1-2020-06-22/ja/validated.tsv'
        original_data_pkl = './CommonVoice/dataset_CommonVoice/read_table.pkl'
        train_data_pkl = './CommonVoice/dataset_CommonVoice/train_dataset.pkl'
        valid_data_pkl = './CommonVoice/dataset_CommonVoice/val_dataset.pkl'

        #データセットの一意性確認と正解ラベルの列挙
        try: 
            with open(original_data_path, "rb") as f1:
                df = pickle.load(f1)
        except :
            import pandas as pd
            df = pd.read_table(original_data_path)
            with open(original_data_pkl, "wb") as f2:
                pickle.dump(df, f2)
        assert not df.path.duplicated().any()
        self.classes = df.client_id.drop_duplicates().tolist()
        self.n_classes = len(self.classes)

        if train==True:
            # f = open (train_data_pkl, 'rb')  ######
            try:
                with open(train_data_pkl, "rb") as f5:
                    self.dataset = pickle.load(f5)
            except:
                data_dirs = original_data_path.split('/')
                dataset = torchaudio.datasets.COMMONVOICE(
                    '/'.join(data_dirs[:-4]), tsv=data_dirs[-1])
                # データセットの分割
                n_train = int(len(dataset) * split_rate)
                n_val = len(dataset) - n_train
                torch.manual_seed(torch.initial_seed())  # 同じsplitを得るために必要
                self.dataset, _ = random_split(dataset, [n_train, n_val])
                with open(train_data_pkl, "wb") as f3:
                    pickle.dump(self.dataset, f3)
            
        elif train==False:
            # f = open (valid_data_pkl, 'rb') ######
            try:
                with open(valid_data_pkl, "rb") as f6:
                    self.dataset = pickle.load(f6)
            except:
                data_dirs = original_data_path.split('/')
                dataset = torchaudio.datasets.COMMONVOICE(
                    '/'.join(data_dirs[:-4]), tsv=data_dirs[-1])
                # データセットの分割
                n_train = int(len(dataset) * split_rate)
                n_val = len(dataset) - n_train
                torch.manual_seed(torch.initial_seed())  # 同じsplitを得るために必要
                _, self.dataset = random_split(dataset, [n_train, n_val])
                with open(valid_data_pkl, "wb") as f4:
                    pickle.dump(self.dataset, f4)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, sample_rate, dictionary = self.dataset[idx]
        #リサンプリングしておくと、以降は共通sample_rateでtransformできる
        if sample_rate != self.sample_rate:
            x = torchaudio.transforms.Resample(sample_rate)(x)
        #各種変換、MFCC等は外部でtransformとして記述
        #ただし、推論と合わせるためにMFCCは先に済ませておく
        x = torchaudio.transforms.MFCC(log_mels=True)(x)
        #最終的にxのサイズを揃える
        if self.transform:
            x = self.transform(x)
        #特徴量：音声テンソル、　正解ラベル：話者IDのインデックス
        return x, self.classes.index(dictionary['client_id'])

class xve
    ctor_Dataset(Dataset):
    def __init__(self, enrollment=True, transform=None):
        self.transform = transform

        self.datasets = []

        if enrollment==True:
            path = os.path.join(ROOT_PATH + 'M00_enr')
            for pathname, dirnames, filenames in os.walk(path):
                for filename in sorted(filenames):
                    wavfile = ROOT_PATH + 'M00_enr/' + filename
                    label = int(filename[4]) - 1
                    data, sr = torchaudio.load(wavfile)
                    data = torch.tensor(data)
                    data = [data, label]
                    self.datasets.append(data)

        if enrollment==False: #testデータ
            for speaker in speakers:
                label = int(speaker[3]) - 1
                path = os.path.join(ROOT_PATH + speaker + 'test')
                for pathname, dirnames, filenames in os.walk(path):
                    for filename in filenames:
                        wavfile = ROOT_PATH + speaker + 'test/' + filename
                        data, sr = torchaudio.load(wavfile)
                        data = torch.tensor(data)
                        data = [data, label]
                        self.datasets.append(data)

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        x, label = self.datasets[idx]
        x = torchaudio.functional.vad(x, 16000)
        x = torch.fliplr(x)
        x = torchaudio.functional.vad(x, 16000)
        x = torch.fliplr(x)
        #推論と合わせるためにMFCCは先に済ませておく
        x = torchaudio.transforms.MFCC(log_mels=True)(x)
        #最終的にxのサイズを揃える
        if self.transform:
            x = self.transform(x)
        #特徴量：音声テンソル、　正解ラベル：話者IDのインデックス
        return x, label
               
#学習モデル
class SpeechNet(nn.Module):
    def __init__(self, n_classes, x_vector=1024):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm1d(40),
            nn.Conv1d(40, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.fc = nn.Sequential(
            nn.Linear(30*64, x_vector),
            nn.BatchNorm1d(x_vector),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(x_vector, n_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

#最後の1次元に指定サイズにCropし、長さが足りないときはCircularPadする
#音声データの時間方向の長さを揃えるために使うtransform部品
class CircularPad1dCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, x):
        n_repeat = self.size // x.size()[-1] + 1
        repeat_sizes = ((1,) * (x.dim() - 1)) + (n_repeat,)
        out = x.repeat(*repeat_sizes).clone()
        return out.narrow(-1, 0, self.size)

#コサイン類似度を求める関数
def cos_similarity(x, y):
    dot = torch.matmul(x, torch.t(y))
    norm = torch.sqrt(torch.matmul(x,torch.t(x))) * torch.sqrt(torch.matmul(y,torch.t(y)))
    return (dot / norm).item()

def SpeechML(train_dataset=None, val_dataset=None, enrollment=None, test_dataset=None, *,
             n_classes=None, n_epochs=15,
             load_pretrained_state=None, test_last_hidden_layer=False,
             show_progress=True, show_chart=False, save_state=False, x_vector=1024):

    #モデルの準備
    if not n_classes:
        assert train_dataset, 'train_dataset or n_classes must be a valid.'
        n_classes = train_dataset.n_classes
        print(f'n_classes:{n_classes}')
    model = SpeechNet(n_classes, x_vector)
    model = model.to(device)
    print(f'x-vector has {x_vector} layers.')

    if load_pretrained_state:
        model.load_state_dict(torch.load(load_pretrained_state))
    criterion = nn.CrossEntropyLoss()
    ##### optimizerは要調整
    #optimizer = torch.optim.Adam(model.parameters())
    #optimizer = torch.optim.Adagrad(model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    #前処理の定義
    Squeeze2dTo1d = lambda x: torch.squeeze(x, -3)
    train_transform = transforms.Compose([
        CircularPad1dCrop(800),
        transforms.RandomCrop((40, random.randint(160, 320))),
        transforms.Resize((40, 240)),
        Squeeze2dTo1d,
    ])
    test_transform = transforms.Compose([
        CircularPad1dCrop(240),
        Squeeze2dTo1d
    ])

    #学習データ・テストデータの準備
    batch_size = 32
    if train_dataset:
        train_dataset.transform = train_transform
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
    if not train_dataset:
        n_epochs = 0  #学習データが無ければエポックはまわせない
    if val_dataset:
        val_dataset.transform = test_transform
        val_dataloader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=True)
    if enrollment:
        enrollment.transform = test_transform
        print("1111111", enrollment.transform)
        print("2222222",enrollment)
        enrollment_dataloader = DataLoader(
            enrollment, batch_size=10, shuffle=False)
    if test_dataset:
        test_dataset.transform = test_transform
        test_dataloader = DataLoader(
            test_dataset, batch_size=10, shuffle=False)
    print(train_dataloader)
    print(val_dataloader)
        
    #学習
    losses = []
    accs = []
    val_losses = []
    val_accs = []
    for epoch in range(n_epochs):
        #学習ループ
        running_loss = 0.0
        running_acc = 0.0
        for x_train, y_train in train_dataloader:
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            optimizer.zero_grad()
            vector = model(x_train)            
            loss = criterion(vector, y_train)
            loss.backward()
            running_loss += loss.item()
            pred = torch.argmax(vector, dim=1)
            running_acc += torch.mean(pred.eq(y_train).float())
            optimizer.step()
        running_loss /= len(train_dataloader)
        running_acc /= len(train_dataloader)
        losses.append(running_loss)
        accs.append(running_acc)
        
        #検証ループ
        val_running_loss = 0.0
        val_running_acc = 0.0
        for val_test in val_dataloader:
            if not(type(val_test) is list and len(val_test)==2):
                break
            x_val, y_val = val_test[0].to(device), val_test[1].to(device)    
            y_pred = model(x_val)
            val_loss = criterion(y_pred, y_val)
            val_running_loss += val_loss.item()
            pred = torch.argmax(y_pred, dim=1)
            val_running_acc += torch.mean(pred.eq(y_val).float())
        val_running_loss /= len(val_dataloader)
        val_running_acc /= len(val_dataloader)
        can_save = (val_running_acc > 0.9 and
                    val_running_loss < min(val_losses))
        val_losses.append(val_running_loss)
        val_accs.append(val_running_acc)
        if show_progress:
            print(f'epoch:{epoch+1}, loss:{running_loss:.3f}, '
                  f'acc:{running_acc:.3f}, val_loss:{val_running_loss:.3f}, '
                  f'val_acc:{val_running_acc:.3f}, can_save:{can_save}')
        
        #あらかじめmodelフォルダを作っておく
        if save_state and can_save: #can_save==Trueのときのモデルを保存
            torch.save(model.state_dict(), f'{model_save_path}/model-epoch{epoch+1:02}.pth')
        if save_state and ((epoch+1)%10==0): #10epochごとに保存
            torch.save(model.state_dict(), f'{model_save_path}/model-epoch{epoch+1:02}.pth')
    if save_state: #最終epochのモデルを保存
        torch.save(model.state_dict(), f'{model_save_path}/model-epoch{epoch+1:02}.pth')

    if n_epochs > 0:
        with open(output_path + "_histories_CV_512.csv", "w", newline="", encoding="utf-16") as f:
            writer = csv.writer(f)
            writer.writerow(losses)
            writer.writerow(accs)
            writer.writerow(val_losses)
            writer.writerow(val_accs)
            f.close()

    #グラフ描画
    if n_epochs > 0 and show_chart:
        fig, ax = plt.subplots(2)
        ax[0].plot(losses, label='train loss')
        ax[0].plot(val_losses, label='val loss')
        ax[0].legend()
        ax[1].plot(accs, label='train acc')
        ax[1].plot(val_accs, label='val acc')
        ax[1].legend()
        plt.show()
        plt.savefig(output_path + 'figure_CV_512.jpg')

    #推論
    if not test_dataset:
        return
    if not enrollment:
        return

    if test_last_hidden_layer:
        model.fc = model.fc[:-1] #最後の隠れ層を出力する
    print("3333333", type(enrollment_dataloader))
    for enrollment in enrollment_dataloader:
        
        x_enr = enrollment[0] if type(enrollment) is list else enrollment
        print("wwwwww",x_enr.shape)
        xvector_enr = model.eval()(x_enr)

    xvector_test = torch.Tensor()
    y_labels = torch.Tensor()
    for test in test_dataloader:
        x_test = test[0] if type(test) is list else test
        y_label = test[1]
        vector = model.eval()(x_test)
        xvector_test = torch.cat([xvector_test, vector])
        y_labels = torch.cat([y_labels, y_label])#正解ラベルの配列

    #similarity_result = torch.zeros([9, 9, 9])
    similarity_result = [[[0 for k in range(EACH_DATA_NUM)] for j in range(TEST_DATA_NUM)] for i in range(ENR_DATA_NUM)]
    for i in range(ENR_DATA_NUM):
        y = xvector_enr[i]
        for j in range(TEST_DATA_NUM):
            for k in range(EACH_DATA_NUM):
                x = xvector_test[j*(EACH_DATA_NUM) + k]
                #similarity_result[i, j, k] = cos_similarity(x, y)
                similarity_result[i][j][k] = cos_similarity(x, y)
    return similarity_result

def EER(cos_similarities):
    t = 0.01
    count_FRR = []
    count_FAR = []
    threshold = []

    for l in range(100):
        threshold.append(l*t)
        frr = 0
        for i in range(ENR_DATA_NUM):
            for j in range(EACH_DATA_NUM):
                if cos_similarities[i][i][j] < threshold[l]:
                    frr += 1
        count_FRR.append(frr / (TEST_DATA_NUM*EACH_DATA_NUM))


        # FAR
        All = 0
        for i in range(ENR_DATA_NUM):
            for j in range(TEST_DATA_NUM):
                for k in range(EACH_DATA_NUM):
                    if cos_similarities[i][j][k] >= threshold[l]:
                        All += 1

        count_FAR.append((All - (TEST_DATA_NUM*EACH_DATA_NUM - frr)) / (ENR_DATA_NUM*(TEST_DATA_NUM-1)*EACH_DATA_NUM))

        if count_FRR[l] >= count_FAR[l]:
            #a1 = (count_FRR[l] - count_FRR[l-1]) / threshold[l] - threshold[l-1]
            #a2 = (count_FAR[l] - count_FAR[l-1]) / threshold[l] - threshold[l-1]
            #b1 = count_FRR[l] - a1*threshold[l]
            #b2 = count_FAR[l] - a2*threshold[l]
            
            #Xeer = (b2 - b1) / (a1 - a2)
            #Yeer = (a1*b2 - a2*b1) / (a1 - a2)

            p = count_FAR[l-1] - count_FRR[l-1]
            q = count_FRR[l] - count_FAR[l]
            r = p / (p + q)

            Xeer = threshold[l-1] + r * (threshold[l] - threshold[l-1])
            Yeer = count_FRR[l-1] + r * (count_FRR[l] - count_FRR[l-1])   

            print(f'Xeer: {Xeer}')
            print(f'Yeer: {Yeer}')
            print(f"Collation rate: {1- Yeer}")
            print_all_result(Xeer, Yeer, cos_similarities)
            return

def print_all_result(x, y, cos_similarities):
    # figureを生成する
    fig = plt.figure()

    # axをfigureに設定する
    ax = fig.add_subplot(1, 1, 1)
    
    ax.set_xlabel("data num")
    ax.set_ylabel("cosine similarity")
    

    cmap = plt.get_cmap("tab10")

    for i in range(len(cos_similarities)):
        x = []
        y = []
        for j in range(ENR_DATA_NUM):
            for k in range(EACH_DATA_NUM):
                x.append((EACH_DATA_NUM)*j+k)
                y.append(cos_similarities[i][j][k])
        ax.scatter(x, y, color=cmap(i), label=f"enr{i+1}")
    lg = plt.legend(loc='upper left', bbox_to_anchor=(1, 1)) # 凡例を表示
    xticks_list = []
    for enr_num in range(ENR_DATA_NUM):
        xticks_list.append((enr_num+1)*EACH_DATA_NUM)
    plt.xticks(xticks_list)
    plt.grid()
    plt.savefig(graph_save_name, 
            dpi=300, 
            format='png', 
            bbox_extra_artists=(lg,), 
            bbox_inches='tight')
   
if __name__ == '__main__':
    train_dataset = TrainDataset(train=True)
    val_dataset = TrainDataset(train=False)
    enrollment = xvector_Dataset(enrollment=True)
    test_dataset = xvector_Dataset(enrollment=False)

    result = SpeechML(train_dataset=train_dataset, val_dataset=val_dataset, enrollment=enrollment, test_dataset=test_dataset, 
        n_classes=None, n_epochs=0 ,
        load_pretrained_state=f"speech_model/model-epoch{EPOCH_NUM}.pth", test_last_hidden_layer=True, show_chart=False, save_state=False, x_vector=512)

    #crate true label list : 本人同士なら1、それ以外は0
    #similarity_true = [[[0 for k in range(10)] for j in range(9)] for i in range(9)]
    #for i in range(9):
    #    for j in range(9):
    #        for k in range(10):
    #            if i == j:
    #                similarity_true[i][j][k] = 1
    #            else:
    #                similarity_true[i][j][k] = 0

    EER(result)

