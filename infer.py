import cv2
import os
import numpy as np
import keras
from keras.applications import VGG16
from keras import backend as K
from keras.models import Model
import sys
import time
import multiprocessing
from termcolor import colored

from sos import alert
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
model = keras.models.load_model('model/vlstm_92.h5')
image_model = VGG16(include_top=True, weights='imagenet')  
model.summary()  
transfer_layer = image_model.get_layer('fc2')
image_model_transfer = Model(inputs=image_model.input,outputs=transfer_layer.output)
transfer_values_size = K.int_shape(transfer_layer.output)[1]

# Frame size  
img_size = 224

img_size_touple = (img_size, img_size)

# Number of channels (RGB)
num_channels = 3

# Flat frame size
img_size_flat = img_size * img_size * num_channels

# Number of classes for classification (Violence-No Violence)
num_classes = 2

# Number of files to train
_num_files_train = 1

# Number of frames per video
_images_per_file = 20

# Number of frames per training set
_num_images_train = _num_files_train * _images_per_file

# Video extension
video_exts = ".avi"

in_dir = "data"

def get_frames(current_dir, file_name):
    in_file = os.path.join(current_dir, file_name)
    images = []
    vidcap = cv2.VideoCapture(in_file)
    success,image = vidcap.read()
    count = 0
    while count<_images_per_file:
        RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        res = cv2.resize(RGB_img, dsize=(img_size, img_size),interpolation=cv2.INTER_CUBIC) 
        images.append(res)
        success,image = vidcap.read()
        count += 1
    resul = np.array(images)
    resul = (resul / 255.).astype(np.float16)
    return resul


def get_transfer_values(current_dir, file_name):
    
    shape = (_images_per_file,) + img_size_touple + (3,)
    image_batch = np.zeros(shape=shape, dtype=np.float16)
    image_batch = get_frames(current_dir, file_name)
    shape = (_images_per_file, transfer_values_size)
    transfer_values = np.zeros(shape=shape, dtype=np.float16)

    transfer_values = \
            image_model_transfer.predict(image_batch)
            
    return transfer_values


def infer(curr_dir,file_name):
    tr = get_transfer_values(curr_dir,file_name)
    print('inp shape = ',tr.shape)
    tr = tr[np.newaxis,...]
    inp = np.array(tr)
    print('inp shape = ',inp.shape)
    pred = model.predict(inp)
    print(pred)
    res = np.argmax(pred[0])
    if res == 0:
        print("\n\n"+ colored('VIOLENT','red')+" Video with confidence: "+str(round(pred[0][res]*100,2))+" %")
        return 'Violent',str(round(pred[0][res]*100,2))
    else:
        print("\n\n" + colored('NON-VIOLENT','green') +" Video with confidence: "+str(round(pred[0][res]*100,2))+" %")
        return 'Non-Violent',str(round(pred[0][res]*100,2))


if __name__ == "__main__":
    arg = sys.argv
    video_name = arg[1]
    start_time = time.time()
    result,confidence =infer(in_dir,video_name)
    if result=='Violent':
        color = (0,0,255)
        alert()
    else:
        color = (0,255,0)

    end_time = time.time()


    delta = round(end_time-start_time,2)
    fps = round(20/(delta*2),2)
    print("Inferrence time: "+str(delta)+" s")
    print(str(fps)+" fps ^_^")
    vpath = in_dir + '\\' + video_name
    cap = cv2.VideoCapture(vpath)
    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        time.sleep(0.05)
        try:
            frame = cv2.putText(frame,result + ' detected', (40,40), cv2.FONT_HERSHEY_SIMPLEX, 1,color, 2, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            cv2.imwrite('result/'+str(result)+str(count)+'.jpg',frame)
            time.sleep(0.05)
            count+=1
        except:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
