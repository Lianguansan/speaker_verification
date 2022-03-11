#print(f'Event: {event}')
text = sg.Text("Speaker verification")
text1 = sg.Text('Please record your voice ( 3 sec.).')
textinput = sg.InputText()
bt = sg.Button('Confirm')
cbt = sg.Button('Cancel')
bt_enr = sg.Button('Entry')
bt_record = sg.Button('Test')
bt_listen = sg.Button('Listen')
image_element = sg.Image(filename='speaker.png')
layout = [[text],[text1,bt_enr,bt_record,bt_listen],[bt, cbt],[image_element]]
window5 = sg.Window('Welcome to Speaker verification system', layout, size=(400, 400))

if event == 'Confirm':
    while True:
        event, values = window5.read()
        print(event,values)
        if event in (None,'Confirm'):
            break      #close the window
        elif event == 'Record':
            #file_number = file_number + 1
            wf.sourcesfile(name,output_test) #test file を保存
        elif event == 'Listen':
            #sd.play(f"{output_test}{name}.wav",fs = 16000)
            playsound(f"C:/LIAN/授業/project/speech2text/test/{name}.wav") #絶対パスにしなければならない？
            #playsound(f"{output_test}{name}.wav")
        elif event == 'Cancel':
            break
    window5.close()