
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchaudio
from torchvision import transforms
# soxはインストールが必要かも
torchaudio.set_audio_backend('soundfile')
model_path = f"speech-model.pth" # 山地さん
threthold = 0.71 # 山地さん

class xvector_Dataset(Dataset):
    def __init__(self, file_path, enrollment=True, transform=None):
        self.transform = transform
        self.datasets = []
        if enrollment==True:
            data, sr = torchaudio.load(file_path)
            data = torch.tensor(data)
            data = [data, 1]
            self.datasets.append(data)

        if enrollment==False: #testデータ
            data, sr = torchaudio.load(file_path)
            data = torch.tensor(data)
            data = [data, 1]
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

def SpeechML(enrollment=None, test_dataset=None, *,
             n_classes=None, n_epochs, load_pretrained_state,x_vector):
    model = SpeechNet(n_classes, x_vector)
    model.load_state_dict(torch.load(load_pretrained_state))
    #前処理の定義
    Squeeze2dTo1d = lambda x: torch.squeeze(x, -3)
    test_transform = transforms.Compose([
        CircularPad1dCrop(240),
        Squeeze2dTo1d
    ])

    enrollment.transform = test_transform
    enrollment_dataloader = DataLoader(
        enrollment, batch_size=10, shuffle=False)

    test_dataset.transform = test_transform
    test_dataloader = DataLoader(
        test_dataset, batch_size=10, shuffle=False)

    model.fc = model.fc[:-1] #最後の隠れ層を出力する
    for enrollment in enrollment_dataloader:
        x_enr = enrollment[0] if type(enrollment) is list else enrollment
        xvector_enr = model.eval()(x_enr)

    xvector_test = torch.Tensor()
    y_labels = torch.Tensor()
    for test in test_dataloader:
        x_test = test[0] if type(test) is list else test
        y_label = test[1]
        vector = model.eval()(x_test)
        xvector_test = torch.cat([xvector_test, vector])
        y_labels = torch.cat([y_labels, y_label])#正解ラベルの配列

    return cos_similarity(xvector_enr[0], xvector_test[0])

def verification(enr_file_path, test_file_path):

    enrollment = xvector_Dataset(file_path=enr_file_path, enrollment=True)
    test_dataset = xvector_Dataset(file_path=test_file_path, enrollment=False)
    cos_sim = SpeechML(enrollment=enrollment, test_dataset=test_dataset, 
        n_classes=170, n_epochs=0 ,
        load_pretrained_state=model_path, x_vector=512)
    if cos_sim > threthold:
        return True
    else:
        return False

if __name__ == '__main__':
    enr_file_path = "our_data16/M00_enr/NM001000_HS.wav"
    test_file_path = "our_data16/M001test/NM001001_HS.wav"
    print(verification(enr_file_path, test_file_path))