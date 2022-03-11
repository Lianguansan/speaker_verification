import PySimpleGUI as sg
import write_file as wf
from playsound import playsound
from verification import verification

things = ["李 瑞峰", "連　冠三", "晝間　大輝", "豊泉 翔太郎", "小林 千真", "山地 陽太"]

layout = [
    [sg.Text("お名前を選んでください.", font=("Noto Serif CJK JP", 15))],
    [sg.InputCombo(things, key="-combo-", size=(15, 1), font=("Noto Serif CJK JP", 15)),
     sg.Button("Ok", font=("Helvetica", 15)),
     sg.Button("Quit", font=("Helvetica", 15))],
    [sg.Text("お名前を入力してください.", font=("Noto Serif CJK JP", 15))],
    [sg.Input(key="-input-", size=(15, 1), font=("Noto Serif CJK JP", 15)),
     sg.Button("Add", key="-add-", font=("Helvetica", 15))],
    [sg.Image(filename='1.png')],
]
window = sg.Window("話者照合システム", layout)

while True:

    event, values = window.read()

    if event == sg.WINDOW_CLOSED or event == "Quit":
        print('キャンセルした')
        break
        window.close()
    elif event == "-add-":  # Correct event to add
        things.append(values["-input-"])
        window['-combo-'].update(values=things)  # Statement to update
        print('You entered ', values["-input-"])
    elif event == "Ok":
        window.close()
        name = values["-combo-"]
        print('ok', name)
        layout = [
            [(sg.Text(name, font=("Noto Serif CJK JP", 15))),
             (sg.Text('さん、次の作業を確認してください.', font=("Noto Serif CJK JP", 15)))],
            [(sg.Button('登録する', font=("Noto Serif CJK JP", 15))),
             (sg.Button('テスト', font=("Noto Serif CJK JP", 15))),
             (sg.Button('Cancel', font=("Noto Serif CJK JP", 15)))],
            [sg.Image(filename='1.png')],
        ]
        window1 = sg.Window("話者照合システム", layout)
        output_test = "test/"
        output_enr = "enr/"
        while True:
            event, values = window1.read()
            if event == 'Cancel' or event == sg.WINDOW_CLOSED:
                print('キャンセルした')
                break

            elif event in (None, '登録する'):

                layout = [
                    [sg.Text("Please record your voice ( 5 sec.).", font=("Noto Serif CJK JP", 15))],
                    [(sg.Button('Record', font=("Noto Serif CJK JP", 15))),
                     (sg.Button('Listen', font=("Noto Serif CJK JP", 15))),
                     (sg.Button('Cancel', font=("Noto Serif CJK JP", 15)))],
                    [sg.Image(filename='1.png')],
                ]
                window2 = sg.Window("話者照合システム", layout)

                while True:
                    event, values = window2.read()
                    if event == 'Cancel' or event == sg.WINDOW_CLOSED:
                        print('キャンセルした')
                        window2.close()
                        break
                    elif event == 'Record':
                        wf.sourcesfile(name, output_enr)
                        window6 = sg.popup('登録完了しました。')
                    elif event == 'Listen':
                        playsound(
                            f"C:/LIAN/授業/project/GUI/enr/{name}.wav")

            elif event in (None, 'テスト'):

                layout = [
                    [sg.Text("Please record your voice ( 5 sec.).", font=("Noto Serif CJK JP", 15))],
                    [(sg.Button('Record', font=("Noto Serif CJK JP", 15))),
                     (sg.Button('Listen', font=("Noto Serif CJK JP", 15))),
                     (sg.Button('Confirm', font=("Noto Serif CJK JP", 15))),
                     (sg.Button('Cancel', font=("Noto Serif CJK JP", 15)))],
                    [sg.Image(filename='1.png')],
                ]
                window3 = sg.Window("話者照合システム", layout)

                while True:
                    event, values = window3.read()
                    if event == 'Cancel' or event == sg.WINDOW_CLOSED:
                        print('キャンセルした')
                        window3.close()
                        break
                    elif event == 'Record':
                        wf.sourcesfile(name, output_test)
                        window6 = sg.popup('テスト完了しました。')
                    elif event == 'Listen':
                        playsound(
                            f"C:/LIAN/授業/project/GUI/test/{name}.wav")
                    elif event in (None, 'Confirm'):

                        text_r = sg.Text("Result of speaker verification:")
                        text_y = sg.Text("You're the person!")
                        text_n = sg.Text("Sorry.")
                        bt_OK = sg.Button('OK')
                        image_element_y = sg.Image(filename='yes.png')
                        image_element_n = sg.Image(filename='no.png')

                        enr_file_path = f"enr/{name}.wav"
                        test_file_path = f"test/{name}.wav"

                        if verification(enr_file_path, test_file_path) == True:  # 本人
                            layout = [[text_r], [text_y], [bt_OK], [image_element_y]]
                        else:  # 本人でない
                            layout = [[text_r], [text_n], [bt_OK], [image_element_n]]
                        window6 = sg.Window('Welcome to Speaker verification system', layout, size=(400, 400))

                        if event == 'Confirm':
                            while True:
                                event, values = window6.read()
                                if event in (None, 'OK'):
                                    break  # close the window
                                print(f'Event: {event}')
                                print(str(values))
                            window6.close()

                        print("話者照合の結果:", verification(enr_file_path, test_file_path))

        window1.close()