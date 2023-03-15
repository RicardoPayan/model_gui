#Bibliotecas
from tkinter import ttk
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque



#running_inference = False

width_camera, height_camera = 600,450

def select_model(src):
    try:
        model = load_model(src)
    except (OSError):
        print(f"[ERROR] No model found at {src}")
        return None
    return model

#Seccion para abrir la camara sin predicir todavia.
def open_camera(capture, model, label_camara, log):
    ok, frame = capture.read() #Leyendo los frames de la camara

    global counter

    global FPS, frames, frames_queue, current_step, total_time, label_probability
    global CLASSES_LIST, IMAGE_HEIGHT, IMAGE_WIDTH, SEQUENCE_LENGTH, MIN_PROB
    global prediction_frame, prediction_label
    global i


    frames += 1
    frame = cv2.flip(frame,1)
    #Convertir imagen de la camara
    opencv_image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGBA)

    #Transformar el ultimo frame en una imagen
    captured_image = Image.fromarray(opencv_image)

    #Convertir imagen capturada en una photoimage para poder usar en tkinter
    photo_image = ImageTk.PhotoImage(image=captured_image)

    #Poner photo image en label de la camara.
    label_camara.photo_image=photo_image
    label_camara.configure(image=photo_image)

    if running_inference:
        print("running_inference")
        if model is None:
            exit(0)

        #Procesar data para el modelo
        resized_frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
        normalized_frame = resized_frame / 255

        frames_queue.append(normalized_frame)

        total_time = frames / FPS
        label_probability = 0.0

        if counter == SEQUENCE_LENGTH:
            counter = 0

            predicted_labels_probs = model.predict(
                np.expand_dims(frames_queue, axis=0), verbose=0)[0]
            predicted_label = np.argmax(predicted_labels_probs)
            label_probability = predicted_labels_probs[predicted_label]
            print(np.expand_dims(frames_queue, axis=0).shape)

            # si no tengo confianza en la prediccion de un step marco (NO DETECTADO)
            if label_probability < MIN_PROB:
                current_step = ""
            else:
                if (current_step != CLASSES_LIST[predicted_label]):
                    current_step = CLASSES_LIST[predicted_label]
                    time_taken = frames / FPS
                    total_time += time_taken
                    frames = 0

                    log(current_step, time_taken, total_time)

        if current_step:
            print(current_step)
            resultado = ttk.Label(prediction_frame, text = current_step + ", accuracy: " + str(label_probability))
            resultado.grid(column=0,row=i)
            resultado2 = ttk.Label(prediction_frame, text = "Total time: " + str(total_time))
            resultado.grid(column=0,row = i + 1)
            i += 1
        else:
            print('no step detected')
            resultado = ttk.Label(prediction_frame,text="no step detected" + ", accuracy: " + str(label_probability) +
                    "\nTotal time: " + str(total_time))
            resultado.grid(column=0,row = i + 1)
            

        if counter > 60:
            return 

    #Actualizar cada 10ms
    label_camara.after(10, lambda : open_camera(capture, model, label_camara, log))

def log(current_step, time_taken, total_time):
    print(current_step, time_taken, total_time)
    return

def real_time(log=lambda *_ : None):
    model = select_model('v2\idles_model2.h5')
    
    if model is None:
        exit(0)

    FPS = int(capture.get(cv2.CAP_PROP_FPS))
    frames = 0
    counter = 0
    frames_queue = deque(maxlen=SEQUENCE_LENGTH)
    current_step = ""
    total_time = 0.0

    while True:

        ok, frame = capture.read()
        frames += 1
        counter += 1
        if not ok:
            break

        frame = cv2.flip(frame, 1)

            
        opencv_image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGBA)
        captured_image = Image.fromarray(opencv_image)
        photo_image = ImageTk.PhotoImage(image=captured_image)
        label_camara.photo_image=photo_image
        label_camara.configure(image=photo_image)
        label_camara.after(10,open_camera)
        
        #Procesar data para el modelo
        resized_frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
        normalized_frame = resized_frame / 255

        frames_queue.append(normalized_frame)

        if counter == SEQUENCE_LENGTH:
            counter = 0

            predicted_labels_probs = model.predict(
                np.expand_dims(frames_queue, axis=0), verbose=0)[0]
            predicted_label = np.argmax(predicted_labels_probs)
            label_probability = predicted_labels_probs[predicted_label]
            print(np.expand_dims(frames_queue, axis=0).shape)

            # si no tengo confianza en la prediccion de un step marco (NO DETECTADO)
            if label_probability < MIN_PROB:
                current_step = ""
            else:
                if (current_step != CLASSES_LIST[predicted_label]):
                    current_step = CLASSES_LIST[predicted_label]
                    time_taken = frames / FPS
                    total_time += time_taken
                    frames = 0

                    log(current_step, time_taken, total_time)
        
        if current_step:
            print(current_step)
            resultado = ttk.Label(prediction_frame,text=current_step)
            resultado.grid(column=0,row=i)
            i+=1
        else:
            print('no step detected')
            resultado = ttk.Label(prediction_frame,text="no step detected")
            resultado.grid(column=0,row=i)
            i+=1

        if counter > 60:
            break

    print('prediciendo')


def start_inference(): 
    global running_inference
    running_inference = True
    print("Starting inference")

def main():
    #Configuracion general de la ventana.
    root = Tk()
    root.title('Prediction')

    #Declarando los frames y widgets
    #Area de la camara.
    content = ttk.Frame(root) #Content contiene todo los elementos de la ventana.
    camara_frame_label = ttk.Label(content,text="Real time model") #Label para titulo
    camara_frame = ttk.Frame(content, borderwidth=5,relief='ridge',width=650,height=500) #Definiendo el frame donde se mostrara la camara
    label_camara = Label(camara_frame, text="Camara aqui") #Label que muestras las imagenes capturadas por la camara

    global prediction_label, prediction_frame

    #Zona de predicciones
    prediction_label = ttk.Label(content,text="Predictions")
    prediction_frame = Frame(content, width=650, height=400,borderwidth=5,relief='ridge')

    #BOTONES
    #Area de la camara
    start_prediction = ttk.Button(content, text='Iniciar', command=start_inference)
    stop_prediction = ttk.Button(content, text='Detener')

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

    model = select_model('realmodel2.h5')

    #Abrir la camara con opencv
    try:
        capture = cv2.VideoCapture(0)
    except (IOError):
        print(f"[ERROR] Can't open webcam.")
        exit(0)

    global running_inference
    running_inference = False

    global counter
    counter = 0

    global CLASSES_LIST, IMAGE_HEIGHT, IMAGE_WIDTH, SEQUENCE_LENGTH, MIN_PROB
    CLASSES_LIST = ["step-1", "step-2", "step-3", "step-4", "finished", "idle"]
    IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
    SEQUENCE_LENGTH = 30
    MIN_PROB = 0.8

    global FPS, frames, frames_queue, current_step, total_time, label_probability
    global i
    #FPS = int(capture.get(cv2.CAP_PROP_FPS))
    i = 0
    FPS = int(60)
    frames = 0
    #counter = 0
    frames_queue = deque(maxlen=SEQUENCE_LENGTH)
    current_step = ""
    total_time = 0.0
    label_probability = 0.0

    open_camera(capture, model, label_camara, log)

    root.mainloop() #Correr la interfaz.

if __name__ == "__main__":
    main()
