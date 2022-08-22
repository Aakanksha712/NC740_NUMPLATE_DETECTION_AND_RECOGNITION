from firebase import firebase

firebase = firebase.FirebaseApplication("https://esp8266-d2b37-default-rtdb.firebaseio.com/",None)


data_pushed = firebase.get('https://esp8266-d2b37-default-rtdb.firebaseio.com/users','')
#print(data_pushed)
keys = data_pushed.keys()
print(keys)

count_max = 0
rfid = ""

for key in keys:
    dict1 = data_pushed[key]
    for i in dict1.values():
        if i>count_max:
            count_max = i
            rfid = key
print(rfid)





