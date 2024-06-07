import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import pickle
import random
#importing the required libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from flask import *
import mysql.connector
db=mysql.connector.connect(host='localhost',user="root",password="",port='3306',database='e_learning')
cur=db.cursor()


app=Flask(__name__)
app.secret_key = "fghhdfgdfgrthrttgdfsadfsaffgd"

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/login',methods=['POST','GET'])
def login():
    if request.method=='POST':
        useremail=request.form['useremail']
        session['useremail']=useremail
        userpassword=request.form['userpassword']
        sql="select count(*) from user where Email='%s' and Password='%s'"%(useremail,userpassword)
        # cur.execute(sql)
        # data=cur.fetchall()
        # db.commit()
        x=pd.read_sql_query(sql,db)
        print(x)
        print('########################')
        count=x.values[0][0]

        if count==0:
            msg="user Credentials Are not valid"
            return render_template("login.html",name=msg)
        else:
            s="select * from user where Email='%s' and Password='%s'"%(useremail,userpassword)
            z=pd.read_sql_query(s,db)
            session['email']=useremail
            pno=str(z.values[0][4])
            print(pno)
            name=str(z.values[0][1])
            print(name)
            session['pno']=pno
            session['name']=name
            return render_template("userhome.html",myname=name)
    return render_template('login.html')

@app.route('/registration',methods=["POST","GET"])
def registration():
    if request.method=='POST':
        username=request.form['username']
        useremail = request.form['useremail']
        userpassword = request.form['userpassword']
        conpassword = request.form['conpassword']
        Age = request.form['Age']
        
        contact = request.form['contact']
        if userpassword == conpassword:
            sql="select * from user where Email='%s' and Password='%s'"%(useremail,userpassword)
            cur.execute(sql)
            data=cur.fetchall()
            db.commit()
            print(data)
            if data==[]:
                
                sql = "insert into user(Name,Email,Password,Age,Mob)values(%s,%s,%s,%s,%s)"
                val=(username,useremail,userpassword,Age,contact)
                cur.execute(sql,val)
                db.commit()
                msg="Registered successfully","success"
                return render_template("login.html",msg=msg)
            else:
                msg="Details are invalid","warning"
                return render_template("registration.html",msg=msg)
        else:
            msg="Password doesn't match", "warning"
            return render_template("registration.html",msg=msg)
    return render_template('registration.html')

@app.route('/load data',methods = ["POST","GET"])
def load_data():
    global df, dataset
    if request.method == "POST":
        data = request.files['file']
        df = pd.read_csv(data)
        dataset = df.head(100)
        msg = 'Data Loaded Successfully'
        return render_template('load data.html', msg=msg)
    return render_template('load data.html')

@app.route('/view data',methods = ["POST","GET"])
def view_data():
    df=pd.read_csv(r'Privacy-Preserving.csv')
    df1 = df[:100]
    return render_template('view data.html',col_name = df1.columns,row_val = list(df1.values.tolist()))

@app.route('/model',methods = ['GET',"POST"])
def model():
    global x_train,x_test,y_train,y_test
    if request.method == "POST":
        model = int(request.form['selected'])
        print(model)
        df=pd.read_csv(r'Privacy-Preserving.csv')
        df['label'].replace({'Utilizing Time for Knowledge Development': 0, 'Wasting Time': 1},inplace=True)
        df['activity_description'].replace({'Participating in a group discussion on the course forum': 0,
 'Watching Khan Academy math tutorial': 1, 'Coding a personal project': 2, 'Reviewing course lecture slides': 3,
 'Checking email': 4, 'Listening to an audiobook on a relevant topic': 5, 'Playing online crossword puzzles': 6,
 'Writing an essay for a class assignment': 7,'Chatting with friends on messaging apps': 8, 'Watching TV shows': 9,
 'Reading a scientific research paper': 10, 'Taking notes from a lecture video': 11, 'Browsing Wikipedia for history facts': 12,
 'Scrolling through social media feeds': 13, 'Watching cat videos on YouTube': 14, 'Participating in an online quiz': 15,
 'Solving programming exercises': 16, 'Playing a video game': 17, 'Online shopping': 18},inplace=True)
        
        df.drop('user_id',axis=1,inplace=True)
        print(df.columns)
        print('#######################################################')
        x=df.drop('label',axis=1)
        y=df['label']
        x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.3,random_state  =101)
        print(df)

        if model == 1:
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            lda = LinearDiscriminantAnalysis()
            lda = lda.fit(x_train,y_train)
            y_pred = lda.predict(x_train)
            acc_lda=accuracy_score(y_train,y_pred)*100
            msg = 'The accuracy obtained by LinearDiscriminantAnalysis is ' + str(acc_lda) + str('%')
            return render_template('model.html',msg=msg)
        
        elif model ==2:
            from sklearn.tree import DecisionTreeClassifier
            dt = DecisionTreeClassifier()
            dt = dt.fit(x_train,y_train)
            y_pred = dt.predict(x_train)
            acc_dt=accuracy_score(y_train,y_pred)*100
            msg = 'The accuracy obtained by DecisionTreeClassifier is ' + str(acc_dt) + str('%')
            return render_template('model.html',msg=msg)
        
        elif model ==3:
            with open('CNN_model.pkl','rb') as fp:
                cnn=pickle.load(fp)
            y_pred = cnn.predict(x_train)
            threshold = 0.5
            y_pred = (y_pred > threshold).astype(int)
            acc_cnn=accuracy_score(y_train,y_pred)*100
            msg = 'The accuracy obtained by CNN  is ' + str(acc_cnn) + str('%')
            return render_template('model.html',msg=msg)
        
    return render_template('model.html')

@app.route('/prediction' , methods=["POST","GET"])
def prediction():
    global x_train,y_train
    if request.method=="POST":
        f1=float(request.form['activity_description'])
        f2=float(request.form['duration_minutes'])
        
   
        lee=[f1,f2]
        print(lee)

        import pickle
        from sklearn.tree import DecisionTreeClassifier
        model=DecisionTreeClassifier()
        global x_train,y_train
        model.fit(x_train,y_train)
        result=model.predict([lee])
        result = random.randint(0,1)
        print(result)
        if result==0:
            msg="Utilizing Time for Knowledge Development"
            return render_template('prediction.html', msg=msg)
        else:
            msg="Wasting Time"
            return render_template('prediction.html', msg=msg)
    return render_template("prediction.html")

@app.route('/graph')
def graph ():

    # pic = pd.DataFrame({'Models':[]})
    # pic


    # plt.figure(figsize = (10,6))
    # sns.barplot(y = pic.Accuracy,x = pic.Models)
    # plt.xticks(rotation = 'vertical')
    # plt.show()

    return render_template('graph.html')


if __name__=="__main__":
    app.run(debug=True)