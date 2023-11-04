
import PySimpleGUI as sg
import os

from objLoader import Model
from main import preview

def create_sub_name(model_path, divide_factor):
    model_name = os.path.basename(model_path).split('.')[0] + f'divided_{divide_factor}.obj'
    model_dir = os.path.dirname(model_path)
    return os.path.join(model_dir, model_name)

def handle_events(event, values):
    global subdivided_last_path
    
    if event == 'CREATE':
        model_path = values['MODEL']
        if not os.path.exists(model_path):
            sg.popup('Model not exist')
        
        model = Model()
        model.load_obj(model_path)
        
        divide_factor = int(values['SUBS'])
        divided_model = model.create_divided_mobius_model(divide_factor)
        
        model_out_path = create_sub_name(model_path, divide_factor)
        divided_model.save_obj(model_out_path)
    
    if event == 'PREVIEW_MODEL':
        model_path = values['MODEL']
        texture_path = values['TEXTURE']
        
        if not os.path.exists(model_path) or not os.path.exists(texture_path):
            sg.popup('Model or Texture not exist, check your inputs')
            return
        
        preview(model_path, texture_path)
        
    if event == 'PREVIEW_SUB':
        model_path = values['MODEL']
        texture_path = values['TEXTURE']
        divide_factor = int(values['SUBS'])
        
        model_path = create_sub_name(model_path, divide_factor)
        print(model_path)
        texture_path = values['TEXTURE']
        
        if not os.path.exists(model_path) or not os.path.exists(texture_path):
            sg.popup('Model or Texture not exist, check your inputs')
            return
        
        preview(model_path, texture_path)

sg.theme('Dark Grey 13')
font = ("Arial", 20)

layout = [
    [sg.Text('Blended Piecewise Mobius Maps Generator', font=font)],
    [sg.Text('Model:'), sg.Input('', key='MODEL', size=(50,0)), sg.FileBrowse()],
    [sg.Text('Texture:'), sg.Input('', key='TEXTURE', size=(50,0)), sg.FileBrowse()],
    [sg.Text('Subdivisions:'), sg.Spin([i for i in range(1,11)], initial_value=5, key='SUBS', size=(10,1))],
    [sg.Button('Create', key='CREATE')],
    [sg.Button('Priview Model', key='PREVIEW_MODEL', size=(15, 1)), sg.Button('Priview Subdivided', key='PREVIEW_SUB', size=(15, 1))]
]

window = sg.Window('Blended Piecewise Mobius Maps', layout, element_justification='c')

while True:
    event, values = window.read()
    if event in (None, 'Exit'):
        break
    handle_events(event, values)
window.close()


