# attendance-marking_FaceID
A simple GUI with and a face recognition project to mark attendance.

Requirements:
  Python
  DLIB
  face_recognition library
  
Change the location of the refrence image folder and execute the script. It use HOG model by default, tweak the libaray's api to use CNN
and if you have NVIDIA Graphics card, install CUDA for faster results and accuracy

Your refrence image folder should have the images of users face (one image from one person) where it's file name should be the name of the person.

After ececuting the script, you will have a tkinter GUI with a single start button,
  Click the stsrt button
  Which will start the main code to execute and opens your default camera module and will capture 30 frames and finds the faces in it.
  Then it chooses which frame has highest face count and uses it to recognize faces.
  Once the recognition finishes, you can see a image with bounding boxes with the face count and recognized face count and a list box wich contains the names of the recognized peopple.
  And you will have a option called "Check faces" which will show you the detected faces and their names. If the user is not recognized then it will show "Unknown".
  
This is project id done by me and my partner <a href="https://github.com/nikhilSolomon">Nikhil Solomon P</a>, we bith are first year CSE students. This project is not a fully functional software but it can do pretty good work.😅😅
