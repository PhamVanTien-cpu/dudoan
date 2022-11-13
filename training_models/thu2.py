from tkinter import *
from tkinter.ttk import *
from tkinter import messagebox
import tkinter
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import decomposition
from sklearn import datasets

data = pd.read_csv('TongHop1.csv') 

data_Train, data_Test = train_test_split(data, test_size=0.3 , shuffle = False) 
#print('data_Train',data_Train)
#print('data_Test',data_Test)
k = 10
kf = KFold(n_splits=k, random_state=None) 
 
min = 0
i=1

for train_index, test_index in kf.split(data_Train):  
    X_train, X_test = data_Train.iloc[train_index,:-1], data_Train.iloc[test_index, :-1]
   # print('x',test_index) 
    y_train, y_test = data_Train.iloc[train_index, -1], data_Train.iloc[test_index, -1]
    
    SVM = svm.SVC()
    SVM.fit(X_train,y_train) 
           
    Y_pred_train=SVM.predict(X_train) 
    Y_pred_test=SVM.predict(X_test)
    #print('test1',y_test)
    acc = accuracy_score(Y_pred_test, y_test)
    print(" độ chính xác của các lần lặp là:", i,":", acc)
    if(acc > min):
        min = acc
        lan=i 
        good_svm = SVM.fit(X_train, y_train)  
        y_t = y_test
        y_pt = Y_pred_test
    i =i+1;
#print('min',min)
    
print("\n Mo hinh tot nhat",lan)

score = accuracy_score(y_t, y_pt) * 100
#print('test',y_test)

print('==> Tỉ Lệ Chính Xác Là :',score,'%')

print('==> Mô hình tốt nhất Là : ',lan,'\n')

y_predict=good_svm.predict(data_Test.iloc[:,:-1]) 

y_tt = np.array(data_Test.iloc[:,-1])

count = 0
for j in range(len(y_tt)):
     if(y_tt[j] == y_predict[j]):
        count = count + 1
print('\nTỉ lệ dự đoán như sau : ')
print("Số dự đoán đúng :", count, "trên tổng", len(y_tt))
#print("Du Doan :",  good_svm.predict([[67,1,23,78,120,5.6,5.2,1.7,0,0]]))

import pickle
pickle.dump(good_svm, open('E:\web\Test_web\models\model.pkl','wb'))

model = pickle.load(open('E:\web\Test_web\models\model.pkl','rb'))
print("Du Doan :",  model.predict([[67,1,23,78,120,5.6,5.2,1.7,0,0]]))

'''
def showPredict():
    if(txttuoi.get() == "" or txtgioitinh.get() == "" or txtchisokhoi.get() == "" or txtnhiptim.get() == "" or txthuyetap.get() == "" or txtduongmau.get() == ""
       or txtcholesterol.get() == "" or txttriglycerid.get() == ""  or txttiensubenh.get() == "" or txtRANKIN.get() == ""):
         messagebox.showerror("Lỗi Thiếu Thông Tin ", " Vui lòng điền đầy đủ thông tin.");
         
    x_input = np.array([float(txttuoi.get()), float(txtgioitinh.get()), float(txtchisokhoi.get()), float(txtnhiptim.get()), float(txthuyetap.get()), float(txtduongmau.get()),
                        float(txtcholesterol.get()), float(txttriglycerid.get()), float(txttiensubenh.get()), float(txtRANKIN.get())]).reshape(1, -1)
     
#    K_test = last.predict(x_input)
#    print('ktest',K_test)
#    y_dd = good_svm.predict(K_test) 
    
    y_dd = good_svm.predict(x_input) 
    print('Kết Quả :', y_dd[0])
    print('input :', x_input)

    if(y_dd == 1):
        messagebox.showinfo("Kết quả dự đoán: ", "Bạn Có Nguy Cơ Bị Đột Qụy " )
    else :
        messagebox.showinfo("Kết quả dự đoán: ", "Sức Khỏe Bình Thường " )
 #+ str(y_dd[0])

def dochinhxac():
    
    messagebox.showinfo("Khả năng dự đoán của SVM", "Độ chính xác của phương pháp: " + str(accuracy_score(y_tt, y_predict)*100) + "%") 



#khởi tạo cửa số giao diện
windown = Tk()
windown.geometry("750x500")
windown.title("DỰ ĐOÁN NGUY CƠ ĐỘT QUỴ")

#Tạo thông số
lbltuoi = tkinter.Label (windown, text =("Tuổi"), font = ("Arial",10))
lbltuoi.grid(column = 1, row = 2)
txttuoi = Entry(windown, width = 25)
txttuoi.grid(column = 2, row = 2)

lblgioitinh = tkinter.Label (windown, text =("Giới Tính"), font = ("Arial",10))
lblgioitinh.grid(column = 1, row = 3)
txtgioitinh = Entry(windown, width = 25)
txtgioitinh.grid(column = 2, row = 3)

lblchisokhoi = tkinter.Label (windown, text =("Chỉ Số Khối"), font = ("Arial",10))
lblchisokhoi.grid(column = 1, row = 4)
txtchisokhoi = Entry(windown, width = 25)
txtchisokhoi.grid(column = 2, row = 4)

lblnhiptim = tkinter.Label (windown, text =("Nhịp Tim "), font = ("Arial",10))
lblnhiptim.grid(column = 1, row = 5)
txtnhiptim = Entry(windown, width = 25)
txtnhiptim.grid(column = 2, row = 5)

lblhuyetap = tkinter.Label (windown, text =("Huyết Áp"), font = ("Arial",10))
lblhuyetap.grid(column = 1, row = 6)
txthuyetap = Entry(windown, width = 25)
txthuyetap.grid(column = 2, row = 6)

lblduongmau = tkinter.Label (windown, text =("Đường Máu"), font = ("Arial",10))
lblduongmau.grid(column = 3, row = 2)
txtduongmau = Entry(windown, width = 25)
txtduongmau.grid(column = 4, row = 2)

lblcholesterol = tkinter.Label (windown, text =("Cholesterol"), font = ("Arial",10))
lblcholesterol.grid(column = 3, row = 3)
txtcholesterol = Entry(windown, width = 25)
txtcholesterol.grid(column = 4, row = 3)

lbltriglycerid = tkinter.Label (windown, text =("Triglycerid"), font = ("Arial",10))
lbltriglycerid.grid(column = 3, row = 4)
txttriglycerid = Entry(windown, width = 25)
txttriglycerid.grid(column = 4, row = 4)

lbltiensubenh = tkinter.Label (windown, text =("Tiền Sử Bệnh"), font = ("Arial",10))
lbltiensubenh.grid(column = 3, row = 5)
txttiensubenh = Entry(windown, width = 25)
txttiensubenh.grid(column = 4, row = 5)

lblRANKIN = tkinter.Label (windown, text =("RANKIN"), font = ("Arial",10))
lblRANKIN.grid(column = 3, row = 6)
txtRANKIN = Entry(windown, width = 25)
txtRANKIN.grid(column = 4, row = 6)




#Tạo nút bấm

btketqua = Button (windown, text = "Kết Qủa",command = showPredict)
btketqua.place(x = 75, y = 200)

btdochinhxac = Button (windown, text = "Độ Chính Xác",command = dochinhxac)
btdochinhxac.place(x = 275, y = 200)

btthoat = Button (windown, text = "Thoát", command = exit)
btthoat.place(x = 475, y = 200)

windown.mainloop()

'''
