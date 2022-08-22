'''
CODER ZERO
connect with me at: https://www.youtube.com/channel/UCKipQAvBc7CWZaPib4y8Ajg
How to train custom yolov5: https://youtu.be/12UoOlsRwh8
DATASET: 1) https://www.kaggle.com/datasets/deepakat002/indian-vehicle-number-plate-yolo-annotation
         2) https://www.kaggle.com/datasets/elysian01/car-number-plate-detection
'''
### importing required libraries
import torch
import cv2
import time
# import pytesseract
import re
import numpy as np
import easyocr
import pytesseract
from pytesseract import Output
import imutils
from pylab import rcParams
import datetime
import os
import cvzone
from cv2 import dnn_superres

from firebase import firebase

firebase = firebase.FirebaseApplication(
    "https://nc740-number-crunchers-default-rtdb.firebaseio.com/", None)


global original_number
global keyval

global count
count = 0

global curr_time
curr_time = 0


global num_plate_detected
num_plate_detected = []

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
##### DEFINING GLOBAL VARIABLE
EASY_OCR = easyocr.Reader(['en'])  # initiating easyocr
OCR_TH = 0.2


### -------------------------------------- function to run detection ---------------------------------------------------------
def detectx(frame, model):
    frame = [frame]
    print(f"[INFO] Detecting. . . ")
    results = model(frame)
    # results.show()
    # print( results.xyxyn[0])
    # print(results.xyxyn[0][:, -1])
    # print(results.xyxyn[0][:, :-1])

    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    return labels, cordinates


### ------------------------------------ to plot the BBox and results --------------------------------------------------------
def plot_boxes(results, frame, classes):

    global count

    """
    --> This function takes results, frame and classes
    --> results: contains labels and coordinates predicted by model on the given frame
    --> classes: contains the strting labels
    """
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    print(f"[INFO] Total {n} detections. . . ")
    print(f"[INFO] Looping through all detections. . . ")

    ### looping through the detections
    for i in range(n):
        row = cord[i]

        if row[4] >= 0.94:  # threshold value for detection. We are discarding everything below this value
            print(f"[INFO] Extracting BBox coordinates. . . ")
            global count
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fm = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
            print(row[4])
            print(fm)
            count += 1

            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(
                row[2]*x_shape), int(row[3]*y_shape)  # BBOx coordniates
            text_d = classes[int(labels[i])]
            # cv2.imwrite("./output/dp.jpg",frame[int(y1):int(y2), int(x1):int(x2)])

            coords = [x1, y1, x2, y2]

            nplate = getnplate(frame, coords)

            img_dim = nplate.shape

            if (img_dim[1] < img_dim[0]):
                nplate_1 = cv2.rotate(nplate, cv2.cv2.ROTATE_90_CLOCKWISE)
                ocr_result_nplate1 = recognize_plate_easyocr(
                    img=nplate_1, coords=coords, reader=easyocr.Reader(['en']), region_threshold=OCR_TH)
                nplate_2 = cv2.rotate(nplate, cv2.cv2.ROTATE_270_CLOCKWISE)
                ocr_result_nplate2 = recognize_plate_easyocr(
                    img=nplate_2, coords=coords, reader=easyocr.Reader(['en']), region_threshold=OCR_TH)
                plate_len_1 = filter_text(
                    region=nplate_1, ocr_result=ocr_result_nplate1, region_threshold=OCR_TH)
                plate_len_2 = filter_text(
                    region=nplate_2, ocr_result=ocr_result_nplate2, region_threshold=OCR_TH)
                if (plate_len_1 >= plate_len_2):
                    frame = nplate_1
                else:
                    frame = nplate_2

            plate_num = recognize_plate_easyocr(
                img=frame, coords=coords, reader=easyocr.Reader(['en']), region_threshold=OCR_TH)

            print("Confidance : ")
            print(round(float(row[4]), 2))
            # if text_d == 'mask':
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # BBox
            # for text label background
            cv2.rectangle(frame, (x1, y1-20), (x2, y1), (0, 255, 0), -1)
            cv2.putText(frame, f"{round(float(row[4]),2)}", (
                x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # cv2.imwrite("./output/np.jpg",frame[int(y1)-25:int(y2)+25, int(x1)-25:int(x2)+25])

    return frame

###--------------------------------

### --------------------------------ZOOM FUNCTION==============================================================


def zoom(img, zoom_factor=10):
    return cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)

#### ---------------------------- function to recognize license plate --------------------------------------

####===========================================Get nplate======================================================


def getnplate(img, coords):

    xmin, ymin, xmax, ymax = coords
    # get the subimage that makes up the bounded region and take an additional 5 pixels on each side
    nplate = img[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]

    return nplate


###-------------------------------------------------------------------------------------------------------

###-------------------------------------------------------------------------------------------------------


# function to recognize license plate numbers using Tesseract OCR
def recognize_plate_easyocr(img, coords, reader, region_threshold):
    # separate coordinates from box
    xmin, ymin, xmax, ymax = coords
    # get the subimage that makes up the bounded region and take an additional 5 pixels on each side
    nplate = img[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]

    im2 = nplate

    #nplate = img[int(ymin):int(ymax), int(xmin):int(xmax)] ### cropping the number plate from the whole image

    #image noise removal
    nplate = cv2.fastNlMeansDenoisingColored(nplate, None, 10, 10, 9, 21)
    sr = dnn_superres.DnnSuperResImpl_create()
    # Read the desired model
    path = "E:\SIH\yolo_deploy_try_1\EDSR_x3.pb"
    sr.readModel(path)

    # Set the desired model and scale to get correct pre- and post-processing
    sr.setModel("edsr", 3)

    # Upscale the image
    nplate = sr.upsample(nplate)
    cv2.imshow("img_only", nplate)
    cv2.waitKey(6000)

    nplate = cv2.cvtColor(nplate, cv2.COLOR_BGR2GRAY)

    # Gaussian blur
    blur = cv2.GaussianBlur(nplate, (5, 5), 1)

    cv2.imshow("img_only", nplate)
    cv2.waitKey(6000)

    #image increase contrast/brightness
    alpha = 1.5  # Contrast control (1.0-3.0)
    beta = 10  # Brightness control (0-100)
    nplate = cv2.convertScaleAbs(nplate, alpha=alpha, beta=beta)

    cv2.imshow("img_only", nplate)
    cv2.waitKey(6000)

   #binarising the image
    #im_gray = cv.imread('image.png', cv.IMREAD_GRAYSCALE)
    #(thresh, im_bw) = cv2.threshold(nplate, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #nplate = cv2.threshold(nplate, thresh, 255, cv2.THRESH_BINARY)[1]

    ocr_result = reader.readtext(nplate)
    #ocr_result = pytesseract.image_to_string(nplate)
    print(ocr_result)
    print(type(ocr_result))

    if (len(ocr_result) == 0):
        print("No result detected")
    text1 = filter_text(region=nplate, ocr_result=ocr_result,
                        region_threshold=region_threshold)

    if len(text1) == 1:
        text1 = text1[0].upper()
    return ocr_result


### to filter out wrong detections

def filter_text(region, ocr_result, region_threshold):

    global curr_time

    global num_plate_detected
    rectangle_size = region.shape[0]*region.shape[1]

    plate = []
    print(f"[INFO] Print OCR Result... ")
    print(ocr_result)

    original_number = ''
    myplate = []

    for result in ocr_result:
        myplate.append(result[1])

    '''for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        
        if length*height / rectangle_size > region_threshold:
            myplate.append(result[1])'''

    # Convert extracted part in upper case and make a valid number plate
    original_num = ''

    for word in myplate:
        word.upper()
        word = remove(word)
        original_num = original_num+word

    for cha in original_num:
        if (cha.isalpha() or cha.isnumeric()):
            original_number = original_number+cha

    print("PRINTING NUMBER PLATE : ")
    print(original_number.upper())
    num_plate_detected.append(original_number.upper())

    ###----------------------------------------------FIREBASE--------------------------------------------------------------

    ###CODE TO BE REPLACED WITH NIRANJAN'S RFID CODE

    '''rfid_data = {
        'rfid': '12345',
        'Timestamp': curr_time

    }

    result_1 = firebase.post(
        'https://mlmodel1-default-rtdb.firebaseio.com/rfid_detection_data/', rfid_data)'''

    ###----------------------------------------------------------------------------------------------------------------------

    return num_plate_detected

    #-----------------------------------------------Compare Plate----------------------------------------------------------------


def upload_final_numplate_on_firebase():

    ###This function will be called after all three frames are processed

    now = datetime.datetime.now()
    print("Current date and time : ")

    curr_time = now.strftime("%Y-%m-%d %H:%M")

    # Dummy database [ Format : Auto-generated key and dictionary - for dictionary, check firebase-rfid.py ]

    dict_rfid = firebase.get(
        'https://nc740-number-crunchers-default-rtdb.firebaseio.com/rfid_data', '')

    '''# stores detected RFIDS[Niranjan's code part]
    dict_rfid_detected = firebase.get(
        'https://mlmodel1-default-rtdb.firebaseio.com/rfid_detection_data', '')

    last_key = list(dict_rfid_detected.keys())[-1]  # get dict list of keys

    # get value of key [Value datatype is dictionary] It is last detected RFID and timestamp
    last_val = dict_rfid_detected.get(last_key)

    # Value of RFID from last_val [ RFID tag value]
    keys_last_rfid = last_val.get('rfid')'''
#+++++++++++++++++++++++++++++++++Get predefined number plate from database++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    data_pushed = firebase.get(
        'https://nc740-number-crunchers-default-rtdb.firebaseio.com/users', '')
#print(data_pushed)
    keys = data_pushed.keys()
    print(keys)

    count_max = 0
    rfid = ""

    for key in keys:
        dict1 = data_pushed[key]
        for i in dict1.values():
            if i > count_max:
                count_max = i
                rfid = key
    print(rfid)

    rfid_data_values = dict_rfid.values()

    dict_rfid_data = list(rfid_data_values)[0]

    # get corresponding predefined Numberplate value
    val_last_rfid = dict_rfid_data.get(rfid)
    print("**************************************")
    print(val_last_rfid)

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    max_percent = 0  # Variable for percentage
    # Storage of value [ At end, It will have final numberplate value to be uploaded]
    final_num = ''
    for plates in num_plate_detected:

        print(plates)

        set_plates = set(plates)  # convert string into set
        set_keys_last_rfid = set(val_last_rfid)  # convert string into set
        # get no of characters matching
        match_no = len(list(set_plates & set_keys_last_rfid))
        print(match_no)

        percent_match = (match_no/len(val_last_rfid)) * 100  # get match percent
        print(percent_match)
        print(len(plates))

        # Find out maximum percentage
        if (max_percent < percent_match):
            max_percent = percent_match
            final_num = plates  # store corresponding numberplate

        # Store final numberplate to database
    numplate_data = {
        'Number_Plate': final_num,
        'Timestamp': curr_time

    }

    result = firebase.post(
        'https://nc740-number-crunchers-default-rtdb.firebaseio.com/number_plate_data/', numplate_data)
    result_1 = firebase.post(
        'https://nc740-number-crunchers-default-rtdb.firebaseio.com/number_plate_percentage/', max_percent)
    result_2 = firebase.post(
        'https://nc740-number-crunchers-default-rtdb.firebaseio.com/last_detected_RFID/', rfid)


    


### ---------------------------------------------- Main function -----------------------------------------------------

def remove(string):
    return string.replace(" ", "")


###------------------------------------------------Space removal-------------------------------------------------------

def main(img_path=None, vid_path=None, vid_out=None):

    print(f"[INFO] Loading model... ")
    ## loading the custom trained model
    # model =  torch.hub.load('ultralytics/yolov5', 'custom', path='last.pt',force_reload=True) ## if you want to download the git repo and then run the detection
    model = torch.hub.load('E:/SIH/yolo_deploy_try_1/yolov5-master', 'custom', source='local',
                           path='E:/SIH/yolo_deploy_try_1/last.pt', force_reload=True)  # The repo is stored locally

    classes = model.names  # class names in string format

    ### --------------- for detection on image --------------------
    if img_path != None:
        print(f"[INFO] Working with image: {img_path}")
        img_out_name = f"./output/result_{img_path.split('/')[-1]}"

        frame = cv2.imread(img_path)  # reading the image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
       # frame = cv2.threshold(frame, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        #blurred = cv2.GaussianBlur(frame, (7, 7), 0)
        #(T, thresh) = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
       # frame = cv2.medianBlur(frame, 3)

        results = detectx(frame, model=model)  # DETECTION HAPPENING HERE

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        frame = plot_boxes(results, frame, classes=classes)

       # cv2.namedWindow("img_only", cv2.WINDOW_NORMAL) ## creating a free windown to show the result
        # creating a free windown to show the result
        cv2.namedWindow("img_only", cv2.WINDOW_NORMAL)

        while True:
            # frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

            cv2.imshow("img_only", frame)
            upload_final_numplate_on_firebase()

            if cv2.waitKey(5) & 0xFF == ord('q'):
                upload_final_numplate_on_firebase()
                print(f"[INFO] Exiting. . . ")

                # if you want to save he output result.
                cv2.imwrite(f"{img_out_name}", frame)

                break

    ### --------------- for detection on video --------------------
    elif vid_path != None:
        print(f"[INFO] Working with video: {vid_path}")

        ## reading the video

        cap = cv2.VideoCapture(vid_path)

        if vid_out:  # creating the video writer if video output path is given

            # by default VideoCapture returns float instead of int
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            #fps=10
            print(fps)
            codec = cv2.VideoWriter_fourcc(*'mp4v')  # (*'XVID')
            out = cv2.VideoWriter(vid_out, codec, fps, (width, height))

        # assert cap.isOpened()

        cv2.namedWindow("vid_out", cv2.WINDOW_NORMAL)
        while (cap.isOpened()):
            # start_time = time.time()
            ret, frame = cap.read()
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fm = cv2.Laplacian(gray_frame, cv2.CV_64F).var()

            if (ret and fm > 1500):
                print(f"[INFO] Working with frame: ")

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = detectx(frame, model=model)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = plot_boxes(results, frame, classes=classes)

                cv2.imshow("vid_out", frame)
                if vid_out:
                    print(f"[INFO] Saving output video. . . ")
                    out.write(frame)
                if (count >= 3):

                    upload_final_numplate_on_firebase()
                    print(num_plate_detected)
                    break

                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break

        print(f"[INFO] Clening up. . . ")
        ### releaseing the writer
        out.release()

        ## closing all windows
        cv2.destroyAllWindows()


### -------------------  calling the main function-------------------------------

#E:\SIH\test_videos\test_1.mp4
#C:\Users\Lenovo\Downloads\vid_s.mp4
main(vid_path="E:\\SIH\\test_videos\\videoplayback_Trim.mp4",vid_out="vid_1.mp4")  # for custom video
#url = "http://192.168.74.125:8080/video"

#main(vid_path=url,vid_out="webcam_facemask_result.mp4") #### for webcam

#C:\\Users\\Lenovo\\Downloads\\vid_s1.mp4
#E:\\SIH\\Database\\images\\truck30.jpeg
#C:\\Users\\Lenovo\\Downloads\\test_ch_images\\t43.jpg
#C:\\Users\\Lenovo\\Downloads\\test_ch_images\\t49.jpg
#C:\\Users\\Lenovo\\Downloads\\test_ch_images\\t54.jpg

#C:\\Users\\Lenovo\\Downloads\\test_ch_images\\truck11.jpg
#C:\\Users\\Lenovo\\Downloads\\test_ch_images\\truck12.jpg
#C:\\Users\\Lenovo\\Downloads\\test_ch_images\\truck13.jpg
#C:\\Users\\Lenovo\\Downloads\\test_ch_images\\truck14.jpg
#C:\\Users\\Lenovo\\Downloads\\test_ch_images0\\truck15.jpg
#C:\\Users\\Lenovo\\Downloads\\test_ch_images\\truck16.jpg

#main(img_path="E:\\SIH\\number_plate_detection\\number_plate_data\\train\\images\\t15.png") ## for image
