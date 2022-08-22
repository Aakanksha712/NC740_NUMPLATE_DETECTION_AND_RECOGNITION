from firebase import firebase

firebase = firebase.FirebaseApplication("https://mlmodel1-default-rtdb.firebaseio.com/",None)

now = datetime.datetime.now()
print ("Current date and time : ")
    
curr_time = now.strftime("%Y-%m-%d %H:%M:%S")
data = {
        'Number_Plate':,
        'Timestamp':curr_time

    }
    
result = firebase.post('https://mlmodel1-default-rtdb.firebaseio.com/rfid_data/',data)