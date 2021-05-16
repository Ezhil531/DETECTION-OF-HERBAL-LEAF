# main.py
import os
import base64
from flask import Flask, render_template, Response, redirect, request, session, abort, url_for
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.fernet import Fernet
import cv2
import shutil
from flask import send_file
import time
import PIL.Image
from PIL import Image, ImageChops
import numpy as np
import argparse
import imagehash
import mysql.connector
import urllib.request
import urllib.parse
from werkzeug.utils import secure_filename
import extract_vein

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  charset="utf8",
  database="plant_identification"

)

UPLOAD_FOLDER = 'static/trained'
ALLOWED_EXTENSIONS = { 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.secret_key = 'abcdef'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/login', methods=['GET', 'POST'])
def login():
    msg=""
    
    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM admin WHERE username = %s AND password = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            return redirect(url_for('upload'))
        else:
            # Account doesnt exist or username/password incorrect
            msg = 'Incorrect username/password!'
    
        
        
    return render_template('login.html',msg=msg)



##@app.route('/upload', methods=['GET', 'POST'])
##def upload():
##    msg=""
##    
##    if request.method=='POST':
##        plant = request.form['plant']
##        
##
##        mycursor = mydb.cursor()
##        mycursor.execute("SELECT max(id)+1 FROM plant")
##        maxid = mycursor.fetchone()[0]
##        if maxid is None:
##            maxid=1
##        imgname="image"+str(maxid)+".jpg"
##
##
##        if 'file' not in request.files:
##            flash('No file part')
##            return redirect(request.url)
##        file = request.files['file']
##        # if user does not select file, browser also
##        # submit an empty part without filename
##        if file.filename == '':
##            flash('No selected file')
##            return redirect(request.url)
##        if file and allowed_file(file.filename):
##            filename = secure_filename(imgname)
##            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
##
##            timg="train"+str(maxid)+".jpg"
##            shutil.copy("static/upload/"+imgname, 'static/trained/'+timg)
##            
##            sql = "INSERT INTO plant(id, plant, imgname) VALUES (%s, %s, %s)"
##            val = (maxid, plant, imgname)
##            mycursor.execute(sql,val)
##            mydb.commit()
##            msg="Upload success"
##            return redirect(url_for('admin_home', msg=msg))
##    
##    return render_template('upload.html', msg=msg)

   

def getbox(im, color):
    bg = Image.new(im.mode, im.size, color)
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    return diff.getbbox()

def split(im):
    retur = []
    emptyColor = im.getpixel((0, 0))
    box = getbox(im, emptyColor)
    width, height = im.size
    pixels = im.getdata()
    sub_start = 0
    sub_width = 0
    offset = box[1] * width
    for x in range(width):
        if pixels[x + offset] == emptyColor:
            if sub_width > 0:
                retur.append((sub_start, box[1], sub_width, box[3]))
                sub_width = 0
            sub_start = x + 1
        else:
            sub_width = x + 1
    if sub_width > 0:
        retur.append((sub_start, box[1], sub_width, box[3]))
    return retur

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method=='POST':
        #print("d")
        return redirect(url_for('DCNN_train', act="on", page='0', imgg='0'))
    return render_template('upload.html')


@app.route('/DCNN_train', methods=['GET', 'POST'])
def DCNN_train():
    act="on"
    page="0"
    pg=""
    fn=""
    fnn=""
    imgg='1'
    tit=""
    m=0
    n=0
    
    tot=5
    
    
    #if request.method=='POST':
        
        #return redirect(url_for('training', act="on"))
    if request.method=='GET':
        act = request.args.get('act')
        page = request.args.get('page')
        imgg = request.args.get('imgg')
        n = int(page)
        if n==0:
            m = int(imgg)+1
        else:
            m = int(imgg)
            
        pg=str(n)
        page=pg
        imgg = str(m)
        
        mg = m-1
        
        fn="r"+str(m)+".jpg"
        
        if m<=tot:
            act="on"
            
            if n<5:
                if n==0:
                    tit="Preprocessing"
                    image = PIL.Image.open("static/dataset/"+fn)
                    #new_image = PIL.image.resize((300, 300))
                    image.save('static/trained/'+fn)
                    
                    
                    path='static/trained/'+fn
                    im = Image.open(path)

                    pfn="p"+fn
                    path3="static/trained/"+pfn
                    for idx, box in enumerate(split(im)):
                        im.crop(box).save(path3.format(idx))
                    
                    fnn=fn
                elif n==1:
                    tit="Grayscale"
                    pfn="p"+fn
                    path3="static/trained/"+pfn
                    image = Image.open(path3).convert('L')
                    image.save(path3)
                    fnn=pfn

                   
                elif n==2:
                    tit="Segmentation"
                    pfn="p"+fn
                    img = cv2.imread('static/trained/'+pfn)
                    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

                    # noise removal
                    kernel = np.ones((3,3),np.uint8)
                    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

                    # sure background area
                    sure_bg = cv2.dilate(opening,kernel,iterations=3)

                    # Finding sure foreground area
                    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
                    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

                    # Finding unknown region
                    sure_fg = np.uint8(sure_fg)
                    segment = cv2.subtract(sure_bg,sure_fg)
                    fname="s"+fn
                    cv2.imwrite("static/trained/"+fname, segment)
                    fnn=fname
                    
                elif n==3:
                    tit="Feature Selection"
                    image = cv2.imread("static/trained/"+fn)
                    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    #edged = cv2.Canny(gray, 50, 100)
                    fname2="p"+fn
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    img_gaussian = cv2.GaussianBlur(gray,(3,3),0)

                    #canny
                    img_canny = cv2.Canny(image,50,100)
                    
                    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
      
                    # Calcution of Sobelx 
                    sobelx = cv2.Sobel(img_gaussian,cv2.CV_64F,1,0,ksize=5) 
                      
                    # Calculation of Sobely 
                    sobely = cv2.Sobel(img_gaussian,cv2.CV_64F,0,1,ksize=5) 
                      
                    # Calculation of Laplacian 
                    laplacian = cv2.Laplacian(image,cv2.CV_64F)
                    
                    cv2.imwrite("static/trained/"+fname2, img_canny)
                    fnn=fname2
                elif n==4:
                    x=1
                    tit="Classification"
                    
                    fnn=fn
                else:
                    
                    tit="Classified"
                    fnn=fn
                n = int(page)+1
                pg=str(n)
                page=pg
            else:
                tit="Classified"
                fnn=fn
                page='0'
                if m==tot:
                    
                    act="ok"
               
        else:
            act="ok"
                
    #return send_file(path, as_attachment=True)
    return render_template('DCNN_train.html',tit=tit, imgg=imgg, page=page, act=act, fn=fnn)

@app.route('/down')
def down():
    path="test.encrypted"
    return send_file(path, as_attachment=True)

@app.route('/test_upload', methods=['GET', 'POST'])
def test_upload():
    if request.method=='POST':
        #print("d")
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        
        file_type = file.content_type
        # if user does not select file, browser also
        # submit an empty part without filename
        tf=file.filename
        ff=open("log.txt","w")
        ff.write(tf)
        ff.close()
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            fname = "m1.jpg"
            filename = secure_filename(fname)
                
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('DCNN_test', act="on", page='0', imgg='0'))
    return render_template('test_upload.html')


@app.route('/DCNN_test', methods=['GET', 'POST'])
def DCNN_test():
    act="on"
    page="0"
    pg=""
    fn=""
    fnn=""
    imgg='1'
    tit=""
    m=0
    n=0
    
    tot=1
    
    
    #if request.method=='POST':
        
        #return redirect(url_for('training', act="on"))
    if request.method=='GET':
        act = request.args.get('act')
        page = request.args.get('page')
        imgg = request.args.get('imgg')
        n = int(page)
        if n==0:
            m = int(imgg)+1
        else:
            m = int(imgg)
            
        pg=str(n)
        page=pg
        imgg = str(m)
        
        mg = m-1
        
        fn="m1.jpg"
        
        if m<=tot:
            act="on"
            
            if n<5:
                if n==0:
                    tit="Preprocessing"
                    image = PIL.Image.open("static/trained/"+fn)
                    #new_image = PIL.image.resize((300, 300))
                    image.save('static/trained/'+fn)
                    
                    
                    path='static/trained/'+fn
                    im = Image.open(path)

                    pfn="q"+fn
                    path3="static/trained/"+pfn
                    for idx, box in enumerate(split(im)):
                        im.crop(box).save(path3.format(idx))
                    
                    fnn=fn
                elif n==1:
                    tit="Grayscale"
                    pfn="q"+fn
                    path3="static/trained/"+pfn
                    image = Image.open(path3).convert('L')
                    image.save(path3)
                    fnn=pfn

                   
                elif n==2:
                    tit="Segmentation"
                    pfn="q"+fn
                    img = cv2.imread('static/trained/'+pfn)
                    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

                    # noise removal
                    kernel = np.ones((3,3),np.uint8)
                    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

                    # sure background area
                    sure_bg = cv2.dilate(opening,kernel,iterations=3)

                    # Finding sure foreground area
                    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
                    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

                    # Finding unknown region
                    sure_fg = np.uint8(sure_fg)
                    segment = cv2.subtract(sure_bg,sure_fg)
                    fname="s"+fn
                    cv2.imwrite("static/trained/"+fname, segment)
                    fnn=fname
                    
                elif n==3:
                    tit="Feature Selection"
                    image = cv2.imread("static/trained/"+fn)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    edged = cv2.Canny(gray, 50, 100)
                    fname2="f"+fn
                    cv2.imwrite("static/trained/"+fname2, edged)
                    fnn=fname2
                elif n==4:
                    x=1
                    tit="Classification"
                    
                    fnn=fn
                else:
                    
                    tit="Classified"
                    fnn=fn
                n = int(page)+1
                pg=str(n)
                page=pg
            else:
                tit="Classified"
                fnn=fn
                page='0'
                if m==tot:
                    
                    act="ok"
               
        else:
            act="ok"
                
    #return send_file(path, as_attachment=True)
    return render_template('DCNN_test.html',tit=tit, imgg=imgg, page=page, act=act, fn=fnn)

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if request.method=='POST':
        #print("d")
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        
        file_type = file.content_type
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            fname = file.filename
            filename = secure_filename(fname)
                
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('result', act="on", page='0', imgg='0'))
    return render_template('classify.html')


@app.route('/result', methods=['GET', 'POST'])
def result():
    res=""
    password_provided = "xyz" # This is input in the form of a string
    password = password_provided.encode() # Convert to type bytes
    salt = b'salt_' # CHANGE THIS - recommend using a key from os.urandom(16), must be of type bytes
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    key = base64.urlsafe_b64encode(kdf.derive(password)) # Can only use kdf once
    f2=open("log.txt","r")
    vv=f2.read()
    f2.close()
    vv1=vv.split('.')
    tff3=vv1[0]
    tff4=tff3[1:]
    rid=int(tff4)
    input_file = 'static/trained/test.encrypted'
    with open(input_file, 'rb') as f:
        data = f.read()

    fernet = Fernet(key)
    encrypted = fernet.decrypt(data)
    value=encrypted.decode("utf-8")
    dar=value.split('|')
    rr=rid-1
    dv=dar[rr]
    drw=dv.split('-')
    v=int(drw[1])
    if v<=20:
        lf="Oregano"
    elif v<=40:
        lf="Mint"
    else:
        lf="Betel"
    res=lf
    return render_template('result.html',res=res)


@app.route('/logout')
def logout():
    # remove the username from the session if it is there
    session.pop('username', None)
    return redirect(url_for('index'))




if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
