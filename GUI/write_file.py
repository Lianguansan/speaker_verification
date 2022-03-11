import pyaudio
import numpy as np
import soundfile as sf

def record(index, samplerate, fs, time):
    pa = pyaudio.PyAudio()
    data = []
    dt = 1 / samplerate

    # ストリームの開始
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=samplerate,
                     input=True, input_device_index=index, frames_per_buffer=fs)

    # フレームサイズ毎に音声を録音していくループ
    for i in range(int(((time / dt) / fs))):
        frame = stream.read(fs)
        data.append(frame)

    # ストリームの終了
    stream.stop_stream()
    stream.close()
    pa.terminate()

    # データをまとめる処理
    data = b"".join(data)

    # データをNumpy配列に変換
    data = np.frombuffer(data, dtype="int16") / float((np.power(2, 16) / 2) - 1)

    return data, i

def sourcesfile(name,output_path):
    time = 3            # 計測時間[s]
    samplerate = 16000  # サンプリングレート
    fs = 1024           # フレームサイズ
    index = 1           # マイクのチャンネル指標

    wfm, i = record(index, samplerate, fs, time)
    t = np.arange(0, fs * (i + 1) * (1 / samplerate), 1 / samplerate)
    #print(' *recording ')

    #output_dir = f"C:/LIAN/授業/project/speech2text/ourdata_16/M00"+str(file_number)+'test'
    wave_name = f"{output_path}{name}.wav"
    print(wave_name)
    #enr_path =
    sf.write(wave_name, wfm, samplerate, format="WAV", subtype="PCM_16")
    #sf.write("C:/LIAN/授業/project/speech2text/record/record.wav", wfm, samplerate, format="WAV", subtype="PCM_16")

    #print(' *recording done ')

if __name__ == '__main__':
    sourcesfile()
