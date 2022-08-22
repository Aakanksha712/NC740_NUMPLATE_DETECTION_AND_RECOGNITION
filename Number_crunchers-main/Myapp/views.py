import re
from django.shortcuts import render
import pyrebase

from django.views.generic import View
   
from rest_framework.views import APIView
from rest_framework.response import Response
 
 


config = {
  "apiKey": "AIzaSyBEuJVrV_FgvVU4VbuzEt_t8FR7THa6tuw",
  "authDomain": "nc740-number-crunchers.firebaseapp.com",
  "databaseURL": "https://nc740-number-crunchers-default-rtdb.firebaseio.com",
  "projectId": "nc740-number-crunchers",
  "storageBucket": "nc740-number-crunchers.appspot.com",
  "messagingSenderId": "126751447418",
  "appId": "1:126751447418:web:579b3269eb42493d6d1ec3",
  "measurementId": "G-365FVJNYMY"
}
# Initialising database,auth and firebase for further use
firebase=pyrebase.initialize_app(config)
authe = firebase.auth()
database=firebase.database()

def signIn(request):
    return render(request, 'signIn.html')

def admin_signIn(request):
    return render(request, 'admin_signIn.html')

def home(request):
    return render(request,"index.html")

def postsignIn_admin(request):
    email=request.POST.get('email')
    passw = request.POST.get("pass")

    users = database.child(0).child('Details').get()
    li = []
    #https://www.youtube.com/watch?v=sVwWEoDa_uY
    for user in users.each():
        print(user.val())
        li.append(user.val())
    context = {
        "e" : email,
        "d" : li
    }
    
    try:
        #user = authe.sign_in_with_email_and_password(email,passw)
        if (email == "shruti@gmail.com") and (passw == "Shruti"):
            return render(request, "a_wel.html", context)
    except:
        message="invalid credentials"
        return render(request,"admin_signIn.html",{"messg":message})
    return render(request,"admin_signIn.html")
    #print(user['idToken'])
    
 
def postsignIn(request):
    email=request.POST.get('email')
    passw = request.POST.get("pass")

    #a_email=request.POST.get('a_email')
    #a_passw = request.POST.get("a_pass")
    #print(a_email)
    #print(a_passw)
    
    try:
        user = authe.sign_in_with_email_and_password(email,passw)
    except:
        message="invalid credentials"
        return render(request,"signIn.html",{"messg":message})
    #print(user['idToken'])
    return render(request, "wel.html",{"e":email})
 

def comparison(request):
    num_perc = database.child('number_plate_percentage').get()
    num_data = database.child('number_plate_data').get()
    last_Data = database.child('last_detected_RFID').get()
    

    #print(original)

    li = []
    data = []
    last_rfid = []

    for i in num_perc.each():
        li.append(i.val())

    for i in num_data.each():
        data.append(i.val())

    for i in last_Data.each():
        last_rfid.append(i.val())


    
    
    perc = round(li[-1],2)
    extract = data[-1]['Number_Plate']
    rfid = last_rfid[-1]

    original = database.child('rfid_data').child("-NA3ZgJ5oFLd0PJy3-jW").child(rfid).get().val()
    


    context={
        "perc": perc,
        "extract":extract,
        "original": original

    }
    return render(request,"compare.html",context)


def logout(request):
    try:
        del request.session['uid']
    except:
        pass
    return render(request,"index.html")
 
def signUp(request):
    return render(request,"signUp.html")
 
def postsignUp(request):
    email = request.POST.get('email')
    passs = request.POST.get('pass')
    name = request.POST.get('name')
    try:
        user=authe.create_user_with_email_and_password(email,passs)
        uid = user['localId']
        data={"name":name,"status":"1"}
        #database.child(“users“).child(uid).child(“details“).set(data)
    except:
        message="Unable to create account try again"
        return render(request,"signUp.html",{"message":message})
    return render(request,"signIn.html")

def reset(request):
    return render(request, "Reset.html")
 
def postReset(request):
    email = request.POST.get('email')
    try:
        authe.send_password_reset_email(email)
        message  = "A email to reset password is successfully sent"
        return render(request, "Reset.html", {"msg":message})
    except:
        message  = "Something went wrong, Please check the email you provided is registered or not"
        return render(request, "Reset.html", {"msg":message})


class ChartData(APIView):
    authentication_classes = []
    permission_classes = []
    
    

    def get(self, request, format = None):
        la = []
        da = []
        for i in range(1,10):
            m = str(i)
            framework = database.child(0).child('Details').child(m).child('vehicle_no').get().val()
            frame = database.child(0).child('Details').child(m).child('KM').get().val()
            la.append(framework)
            da.append(frame)
        print(la)
        print(da)
        labels = la
        chartLabel = "Distance in KM"
        chartdata = da
        data ={
                     "labels":labels,
                     "chartLabel":chartLabel,
                     "chartdata":chartdata,
             }
             
        '''

        labels = [
            'January',
            'February', 
            'March', 
            'April', 
            'May', 
            'June', 
            'July'
            ]
        chartLabel = "my data"
        #labels = ['2021-11-06 23:39:30', '2021-11-07 01:00:28', '2021-11-07 09:00:28', '2021-10-07 09:00:28', '2021-01-17 09:00:28', '2022-10-09 09:00:28', '2020-11-07 09:00:28']
        chartdata = [0, 10, 5, 2, 20, 30, 45]
        data ={
                     "labels":labels,
                     "chartLabel":chartLabel,
                     "chartdata":chartdata,
             }'''
        return Response(data)
