import tkinter as tk
import cv2
from PIL import ImageTk, Image
import time

window = tk.Tk()
canvas = tk.Canvas(window, width=1300, height=750)
canvas.pack()
img = ImageTk.PhotoImage(Image.fromarray(cv2.imread("2012-12-12_10_00_05.jpg", cv2.IMREAD_GRAYSCALE)))
canvas.create_image(10,10, anchor=tk.NW, image=img)
window.mainloop()

time.sleep(2)
img = ImageTk.PhotoImage(Image.fromarray(cv2.imread("2012-09-21_06_10_10.jpg", cv2.IMREAD_GRAYSCALE)))
canvas.create_image(10,10, anchor=tk.NW, image=img)
time.sleep(2)
img = ImageTk.PhotoImage(Image.fromarray(cv2.imread("2012-12-12_10_00_05.jpg", cv2.IMREAD_GRAYSCALE)))
canvas.create_image(10,10, anchor=tk.NW, image=img)
time.sleep(2)
img = ImageTk.PhotoImage(Image.fromarray(cv2.imread("2012-09-21_06_10_10.jpg", cv2.IMREAD_GRAYSCALE)))
canvas.create_image(10,10, anchor=tk.NW, image=img)
time.sleep(2)
img = ImageTk.PhotoImage(Image.fromarray(cv2.imread("2012-12-12_10_00_05.jpg", cv2.IMREAD_GRAYSCALE)))
canvas.create_image(10,10, anchor=tk.NW, image=img)