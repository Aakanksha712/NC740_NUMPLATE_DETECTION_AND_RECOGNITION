from firebase import firebase

firebase = firebase.FirebaseApplication("https://nc740-number-crunchers-default-rtdb.firebaseio.com//",None)

data = {
    '407079104':'WB5IA6412', #VideoTrim
    '14091011104':'OR02BU3389' #Database folder-Aakanksha laptop - first image
}
    
result = firebase.post('https://nc740-number-crunchers-default-rtdb.firebaseio.com//rfid_data/',data)

data_pushed = firebase.get('https://nc740-number-crunchers-default-rtdb.firebaseio.com//rfid_data','')
print(data_pushed)


