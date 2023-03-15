#Bibliotecas
from tkinter import ttk
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
import cv2
from tkinter import filedialog
from keras.models import load_model



predicciones = ['step-1','step-2','step-3'] #Solo para probar como imprimir las predicciones

#Abrir la camara con opencv
capture = cv2.VideoCapture(0)
width_camera, height_camera = 600,450


#Configuracion general de la ventana.
root = Tk()
root.title('Prediction')


#Declarando los frames y widgets
#Area de la camara.
content = ttk.Frame(root) #Content contiene todo los elementos de la ventana.
camara_frame_label = ttk.Label(content,text="Real time model") #Label para titulo
camara_frame = ttk.Frame(content, borderwidth=5,relief='ridge',width=650,height=500) #Definiendo el frame donde se mostrara la camara
label_camara = Label(camara_frame, text="Camara aqui") #Label que muestras las imagenes capturadas por la camara


#Zona de predicciones
prediction_label = ttk.Label(content,text="Predictions")
prediction_frame = Frame(content, width=650, height=400,borderwidth=5,relief='ridge')

#BOTONES
#Area de la camara
start_prediction = ttk.Button(content, text='Iniciar')
stop_prediction = ttk.Button(content, text='Detener')
loadModel_button = ttk.Button(content, text='Cargar modelo', command=loadModel)
loadModel_button.grid(column=0, row=4, pady=5, sticky='ns')

#Area de predicciones
save_csv = Button(content,text='Save',bg='#33b249')
delete = Button(content,text='Delete',bg='#f44336')

#Posicionando elementos
#Camara
content.grid(column=0,row=0,sticky='nsew')
camara_frame_label.grid(column=0,row=0,columnspan=3) #Titulo, con su posicion en columna y filas. columnspan es para que atraviese dos columas y quede en medio de las dos
camara_frame.grid(column=0,row=1,columnspan=3,rowspan=2,padx=5,pady=5,sticky='nsew')
label_camara.grid(row=0,column=0,sticky='nsew') #Posicion de la camara
start_prediction.grid(column=0,row=3,pady=5,sticky='ns')
stop_prediction.grid(column=2,row=3,pady=5,sticky='ns')

#Predicciones
prediction_label.grid(column=3,row=0,columnspan=2)
prediction_frame.grid(column=3,row=1,columnspan=2,rowspan=2,padx=5,pady=5,sticky='nsew')
save_csv.grid(column=3,row=3,padx=5,pady=5,sticky='ns')
delete.grid(column=4,row=3,pady=5,sticky='ns')


#Variables que devuelven una tupla con las filas y columnas de las diferentes secciones.
size_content = content.grid_size()
size_prediction = prediction_frame.grid_size()
size_camara = camara_frame.grid_size()
print(size_camara)

#Reajustando los elementos cuando cambia de size la ventana.
root.columnconfigure(0,weight=1)
root.rowconfigure(0,weight=1)
camara_frame.columnconfigure(0,weight=1)
camara_frame.rowconfigure(0,weight=1)
content.columnconfigure(0,weight=3)
content.columnconfigure(1,weight=3)
content.columnconfigure(2,weight=3)
content.columnconfigure(3,weight=1)
content.columnconfigure(4,weight=1)
content.rowconfigure(1,weight=1)

#Seccion para abrir la camara sin predicir todavia.
def open_camera():
    _,frame = capture.read() #Leyendo los frames de la camara

    #Convertir imagen de la camara
    opencv_image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGBA)

    #Transformar el ultimo frame en una imagen
    captured_image = Image.fromarray(opencv_image)

    #Convertir imagen capturada en una photoimage para poder usar en tkinter
    photo_image = ImageTk.PhotoImage(image=captured_image)

    #Poner photo image en label de la camara.
    label_camara.photo_image=photo_image
    label_camara.configure(image=photo_image)

    #Actualizar cada 10ms
    label_camara.after(10,open_camera)

open_camera()

#Este for es solo de prueba para tener una idea de como imprimir las predicciones cuando se hagan en tiempo real
for i in range(len(predicciones)):
    resultado = ttk.Label(prediction_frame,text=predicciones[i])
    resultado.grid(column=0,row=i)


root.mainloop() #Correr la interfaz.