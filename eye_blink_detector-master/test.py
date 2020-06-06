import cv2, dlib
import numpy as np
from imutils import face_utils
from PIL import Image, ImageGrab 
from tensorflow.keras.models import load_model
from decrypt import decrypt
from gaze_tracking import GazeTracking
from pytrie import StringTrie

IMG_SIZE = (34, 26)

# sentences = ['breathlessness','water','toilet','emergency','problem','yes','no','fine','pain','hungry','movie']
trie = StringTrie()
# for sent in sentences:
#   trie[sent] = sent

def get_sentences(prefix):
  return trie.values(prefix)[:4]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('facepred.dat')

model = load_model('models/model.h5')
model.summary()

def crop_eye(img, eye_points):
  x1, y1 = np.amin(eye_points, axis=0)
  x2, y2 = np.amax(eye_points, axis=0)
  cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

  w = (x2 - x1) * 1.2
  h = w * IMG_SIZE[1] / IMG_SIZE[0]

  margin_x, margin_y = w / 2, h / 2

  min_x, min_y = int(cx - margin_x), int(cy - margin_y)
  max_x, max_y = int(cx + margin_x), int(cy + margin_y)

  eye_rect = np.rint([min_x-5, min_y-5, max_x+5, max_y+5]).astype(np.int)

  eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

  return eye_img, eye_rect

# main
# cap = cv2.VideoCapture(0)
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 600,600)
t_frame = 0
buffer=''
sentence=' '

gaze = GazeTracking()
print(get_sentences(''))
while(True):
  printscreen_pil = ImageGrab .grab(bbox =(600, 300, 1200, 600)) 
  printscreen_numpy = np.array(printscreen_pil)
  img_ori = cv2.cvtColor(printscreen_numpy,cv2.COLOR_BGR2RGB)
  # ret, img_ori = cap.read()

  # if not ret:
  #   break
  # img_ori = cv2.resize(img_ori, dsize=(0, 0), fx=0.5, fy=0.5)
  gaze.refresh(img_ori)
  # tmp = get_sentences(sentence)
  # if gaze.is_left():    
  #   try:
  #     sentence = get_sentences(sentence[1:])[0]
  #     print("SENTENCE ",sentence)
  #   except:
  #     pass
  #   continue
  # if gaze.is_right():
  #   try:
  #     sentence = get_sentences(sentence[1:])[0]
  #     print("SENTENCE ",sentence)
  #   except:
  #     pass
  #   continue
  # if gaze.is_up():
  #   try:
  #     sentence = get_sentences(sentence[1:])[0]
  #     print("SENTENCE ",sentence)
  #   except:
  #     pass
  #   continue
  # if gaze.is_right():
  #   try:
  #     sentence = get_sentences(sentence[1:])[0]
  #     print("SENTENCE ",sentence)
  #   except:
  #     pass
  #   continue

  img = img_ori.copy()
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  faces = detector(gray)

  for face in faces:
    shapes = predictor(gray, face)
    shapes = face_utils.shape_to_np(shapes)

    eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42])
    eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48])

    eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
    eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
    eye_img_r = cv2.flip(eye_img_r, flipCode=1)

    cv2.imshow('l', eye_img_l)
    cv2.imshow('r', eye_img_r)

    eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.
    eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.

    pred_l = model.predict(eye_input_l)
    pred_r = model.predict(eye_input_r)

    # visualize
    state_l = 'O' if pred_l > 0.1 else '_'
    state_r = 'O' if pred_r > 0.1 else '_'

    if state_l == '_' and state_r == '_':
      t_frame+=1
    
    if state_l == 'O' and state_r == 'O':
      if t_frame==0:
        pass
      elif t_frame<5:
        print('dot')#,end=' ')
        buffer+='.'     
      else:
        print('dash')#,end=' ')
        buffer+='-'
      t_frame=0

    if state_l == '_' and state_r == 'O':
      if buffer=='':
        if sentence[-1]!=' ':
          sentence+=buffer+' '
          buffer=''
          print("SENTENCE :",sentence)        
      else:
        a = decrypt(buffer)
        if buffer == -1:
          print("INVALID MORSE")
          buffer=''
        else:
          sentence+=a
          buffer=''
          print("SENTENCE :",sentence)
        print("BUFFER :",buffer)
        
    if state_r == '_' and state_l == 'O':
      if buffer!='' or sentence!='':
        sentence=''
        buffer=''
        print("Sentence cleared..")      

    state_l = state_l % pred_l
    state_r = state_r % pred_r

    cv2.rectangle(img, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(255,255,255), thickness=2)
    cv2.rectangle(img, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(255,255,255), thickness=2)

    cv2.putText(img, state_l, tuple(eye_rect_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(img, state_r, tuple(eye_rect_r[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

  cv2.imshow('image', img)
  if cv2.waitKey(1) == ord('q'):
    break
