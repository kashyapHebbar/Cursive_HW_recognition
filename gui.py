import tkinter as tk
from tkinter import BOTTOM, Button, Label, filedialog

import cv2
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model

from deeplearningmodel.ocr import predict_characters


top = tk.Tk()
top.geometry("800x600")
top.title("Handwriting-recognition")
top.configure(background="#CDCDCD")

label = Label(top, background="#CDCDCD", font=("arial", 15, "bold"))
sign_image = Label(top)
model = load_model("handwriting.model")


def classify(file_path: str) -> str:
    image = cv2.imread(file_path)
    if image is None:
        label.configure(foreground="#B00020", text="Could not read image")
        return ""

    text, _, annotated = predict_characters(model, image)
    label.configure(foreground="#011638", text=text or "No text detected")

    cv2.imshow("Image", annotated)
    cv2.waitKey(0)
    return text


def show_classify_button(file_path: str) -> None:
    classify_button = Button(top, text="Classify Image", command=lambda: classify(file_path), padx=10, pady=5)
    classify_button.configure(background="#364156", foreground="white", font=("arial", 10, "bold"))
    classify_button.place(relx=0.79, rely=0.46)


def upload_image() -> None:
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        image_tk = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=image_tk)
        sign_image.image = image_tk
        label.configure(text="")
        show_classify_button(file_path)
    except Exception:
        pass


upload = Button(top, text="Upload an image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156", foreground="white", font=("arial", 10, "bold"))

upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)
heading = Label(top, text="Handwriting-recognition", pady=20, font=("arial", 20, "bold"))
heading.configure(background="#CDCDCD", foreground="#364156")
heading.pack()
top.mainloop()
