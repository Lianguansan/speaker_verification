import PySimpleGUI as sg
import write_file as wf
from playsound import playsound
import sounddevice as sd
from verification import verification
#torchaudio.set_audio_backend('soundfile')

text = sg.Text("Speaker verification")
text1 = sg.Text('Please input your name.')
image_element = sg.Image(filename='speaker.png')
textinput = sg.InputText()
bt = sg.Button('Confirm')
cbt = sg.Button('Cancel')
layout = [[text],[text1,textinput],[bt, cbt],[image_element]]
window1 = sg.Window('Welcome to Speaker verification system', layout, size=(400, 400))

while True:
    event, values = window1.read()
    print(event, values)
    if event in (None, 'Confirm'):
        print("提出した")
        break  # close the window
    elif event == 'Cancel':
        print('キャンセルした')
        break
    #print(f'Event: {event}')
    #print('value', str(values))
    #print('input', textinput)
    #print('You entered ', values)
window1.close()
print('You entered ', values[0])
name = values[0]

text = sg.Text("Speaker verification")
text1 = sg.Text('Please choose entry or test.')
textinput = sg.InputText()
bt = sg.Button('Confirm')
cbt = sg.Button('Cancel')
bt_enr = sg.Button('Entry')
bt_test = sg.Button('Test')
#bt_listen = sg.Button('Listen')
image_element = sg.Image(filename='speaker.png')
layout2 = [[text],[text1,bt_enr,bt_test],[bt, cbt],[image_element]]
window2 = sg.Window('Welcome to Speaker verification system', layout2, size=(400, 400))
output_test = "test/"
output_enr = "enr/"

while True:
    event, values = window2.read()
    print(event,values)
    if event in (None,'Confirm'):
        break      #close the window
    elif event == 'Entry':
        choice = event
    elif event == 'Test':
        choice = event
window2.close()
print("event:",event)
print("choice:",choice)
if choice == 'Entry':
    text = sg.Text("Entry")
    text1 = sg.Text('Please record your voice (3 sec.).')
    textinput = sg.InputText()
    bt = sg.Button('Confirm')
    cbt = sg.Button('Cancel')
    bt_record = sg.Button('Record')
    bt_listen = sg.Button('Listen')
    image_element = sg.Image(filename='enr.png')
    layout = [[text], [text1, bt_record, bt_listen], [bt, cbt], [image_element]]
    window5 = sg.Window('Welcome to Speaker verification system', layout, size=(400, 400))
    while True:
        event, values = window5.read()
        print(event, values)
        if event in (None, 'Confirm'):
            break  # close the window
        elif event == 'Record':
            # file_number = file_number + 1
            wf.sourcesfile(name, output_enr)  # test file を保存[
            window6 = sg.popup('登録完了しました。')
        elif event == 'Listen':
            # sd.play(f"{output_test}{name}.wav",fs = 16000)
            playsound(f"C:/LIAN/授業/project/GUI/enr/{name}.wav")  # 絶対パスにしなければならない？
            # playsound(f"{output_test}{name}.wav")
        elif event == 'Cancel':
            break
    window5.close()

if choice == 'Test' or 'Confirm':
    text = sg.Text("Test")
    text1 = sg.Text('Please record your voice (3 sec.).')
    textinput = sg.InputText()
    bt = sg.Button('Confirm')
    cbt = sg.Button('Cancel')
    bt_record = sg.Button('Record')
    bt_listen = sg.Button('Listen')
    image_element = sg.Image(filename='test.png')
    layout = [[text], [text1, bt_record, bt_listen], [bt, cbt], [image_element]]
    window5 = sg.Window('Welcome to Speaker verification system', layout, size=(400, 400))
    while True:
        event, values = window5.read()
        print(event, values)
        if event in (None, 'Confirm'):
            break  # close the window
        elif event == 'Record':
            # file_number = file_number + 1
            wf.sourcesfile(name, output_test)  # test file を保存
            window6 = sg.popup('テスト完了しました。')
        elif event == 'Listen':
            # sd.play(f"{output_test}{name}.wav",fs = 16000)
            playsound(f"C:/LIAN/授業/project/GUI/test/{name}.wav")  # 絶対パスにしなければならない？
            # playsound(f"{output_test}{name}.wav")
        elif event == 'Cancel':
            break
    window5.close()
print('event for 5',event)
text_r = sg.Text("Result of speaker verification:")
text_y = sg.Text("You're the person!")
text_n = sg.Text("Sorry.")
bt_OK = sg.Button('OK')
image_element_y = sg.Image(filename='yes.png')
image_element_n = sg.Image(filename='no.png')

enr_file_path = f"enr/{name}.wav"
test_file_path = f"test/{name}.wav"

if verification(enr_file_path, test_file_path) == True: #本人
    layout = [[text_r], [text_y],[bt_OK],[image_element_y]]
else: #本人でない
    layout = [[text_r], [text_n], [bt_OK],[image_element_n]]
window6 = sg.Window('Welcome to Speaker verification system', layout, size=(400, 400))

if event == 'Confirm':
    while True:
        event, values = window6.read()
        if event in (None,'OK'):
            break      #close the window
        print(f'Event: {event}')
        print(str(values))
    window6.close()

print("話者照合の結果:",verification(enr_file_path, test_file_path))