from functools import partial
from tkinter import Tk, Canvas, Label, Button, Listbox

import cv2
import face_recognition
import numpy as np
import os
import shutil
import time
import wx
import xlsxwriter as excel
from PIL import Image, ImageDraw, ImageTk
from memory_profiler import profile

root = Tk()
canvas = Canvas(root)
canvas.grid(row=500, column=500)

wx.App(False)
width, height = wx.GetDisplaySize().Get()


########################################################################################################################
def cam():
    camera = cv2.VideoCapture(0)
    count = 0
    frames, best_len = [], []
    face_cascade = cv2.CascadeClassifier("D:\\Programs\\unfinished (FAILED)\\face_rec\\haarcascade_frontalface_default.xml")
    shutil.rmtree("pull")
    os.system("mkdir pull")

    while count < 30:
        ret, img = camera.read()
        frames.append(img)
        image = np.array(img, "uint8")
        # gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(image, 1.3, 5)
        # print(faces, len(faces))
        best_len.append(len(faces))

        i = 0
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = img[y:y + h, x:x + w]
            cv2.imwrite(f"pull/img{i}.jpg", face)
            i += 1

        count += 1
        cv2.imwrite("best_img.jpg", frames[best_len.index(max(best_len))])
    print(f"{max(best_len)} face's found in camera.")


########################################################################################################################
@profile
def main():
    """hiruthic = face_recognition.load_image_file("hiruthic.jpg")
    hiruthic_data = face_recognition.face_encodings(hiruthic)[0]
    nikhil = face_recognition.load_image_file("nikhil.jpg")
    nikhil_data = face_recognition.face_encodings(nikhil)[0]"""

    start = time.time()

    # known_face_encodings = []  # [hiruthic_data, nikhil_data]
    # known_face_names = []  # ["hiruthic", "nikhil"]

    path = "img"  # input("Path: ")

    os.system("mkdir Fdata")
    print(f"[INFO] Started loading face data to memory.")
    encode_start = time.time()

    try:
        for i in os.listdir(path):
            file = face_recognition.load_image_file(os.path.join(path, i))
            file_encoding = face_recognition.face_encodings(file)[0]
            known_face_encodings.append(file_encoding)
            known_face_names.append(i.split(".")[0])

            file = open(f"Fdata/{i.split('.')[0]}.Fdata", 'w')
            file.write(str(file_encoding))
            file.close()
    except:
        print(f"Folder not found: {path}")

    print(f"[INFO] Face encoding finished in {abs(encode_start - time.time())}")

    start_faces = time.time()

    cam()

    src = face_recognition.load_image_file(
        "best_img.jpg")  # face_recognition.load_image_file("C:\\Users\\hiruthicsha\\Downloads\\Telegram Desktop\\team.jpg")
    face_locations = face_recognition.face_locations(src)
    face_encodings = face_recognition.face_encodings(src, face_locations)
    print(f"[INFO] Face located in: {abs(start_faces - time.time())}")

    pil_image = Image.fromarray(src)

    draw = ImageDraw.Draw(pil_image)
    att = open("attendance.txt", 'w')

    recognize_start = time.time()
    i = 0
    recognized_faces = []
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            if name != "Unknown":
                att.write(name + '\n')
                recognized_faces.append(name)
                # shutil.move(f"C:/User/91948/Downloads/Telegram Desktop/pull/img{i}.jpg", "C:/User/91948/Downloads/Telegram Desktop/pull/{name}.jpg")
        i += 1

        draw.rectangle(((left, top), (right, top)), outline=(0, 0, 255), fill=(200, 200, 200), width=1)

        test_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 0), outline=(0, 0, 0), width=1)
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    att.close()
    print(f"[INFO] Recognition finished in: {abs(recognize_start - time.time())}")
    print(f"[INFO] Program executed in: {abs(start - time.time())}")
    del draw
    pil_image.save("Boxed_faces.jpg")
    # pil_image.show()

    return len(face_locations), recognized_faces


def reset_canvas():
    global canvas, root
    for widget in root.winfo_children():
        widget.destroy()


def pull_faces(recognized_faces):
    row, col, img_size, screen_left = 0, 0, 127, width

    print(f"{recognized_faces} face's recognized.")
    reset_canvas()
    for i, name in enumerate(os.listdir("pull")):
        # print(name)
        im = Image.open(f"pull/img{i}.jpg")
        render = ImageTk.PhotoImage(im)
        img = Label(image=render)
        img.image = render

        if screen_left < 127:
            row += 2
            col = 0
            screen_left = width

        img.grid(row=row, column=col)
        try:
            Label(root, text=recognized_faces[i]).grid(row=row + 1, column=col)
        except IndexError:
            Label(root, text="Unknown").grid(row=row + 1, column=col)
        col += 1
        screen_left -= img_size


def gui():
    face_count, names = main()

    file = open("attendance.txt", 'r')
    recognized_faces = file.readlines()

    im = Image.open("Boxed_faces.jpg")
    im.thumbnail((200, 200), Image.ANTIALIAS)
    render = ImageTk.PhotoImage(im)
    img = Label(image=render)
    img.image = render
    img.grid(row=0, column=0)

    Label(root, text=f"Best frame. {face_count} face's found.\n{len(recognized_faces)} recognized").grid(row=1,
                                                                                                         column=0)

    att = Listbox(root)
    for i, name in enumerate(recognized_faces):
        att.insert(i, name)
    att.grid(row=0, column=1)
    pull = partial(pull_faces, recognized_faces)
    Button(root, text="Check faces", command=pull).grid(row=2, column=2)
    return names


names = gui()
root.mainloop()

class_ = open("class.txt", 'r')
class_people = [line.split("\n")[0] for line in class_.readlines()]
workbook = excel.Workbook("att.xlsx")
worksheet = workbook.add_worksheet()

worksheet.write('A1', 'Attendence')
worksheet.write('B1', 'Absentese')

for i in range(len(names)):
    if names[i] in class_people:
        worksheet.write(i+1, 0, class_people[i])
        class_people.pop(class_people.index(names[i]))

for i in range(len(class_people)):
    worksheet.write(i+1, 1, class_people[i])

workbook.close()

fast_cam() # Live feed