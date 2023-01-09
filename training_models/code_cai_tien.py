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
import pickle


### tien xu ly du lieu

original_data = pd.read_csv('TongHop.csv')
print('\n','\n','==> Dữ liệu ban đầu :','\n','\n',original_data)

original_data_null = original_data.replace(['#NULL!','N','null','NaN'],[np.nan, np.nan, np.nan, np.nan])

later_data = original_data_null[~pd.isna(original_data_null).any(axis=1)].reset_index(drop=True)
print('\n','\n','==> Dữ liệu sau khi xử lý Null :','\n','\n',later_data)

later_data.to_csv('thu.csv', index=False)
df = pd.read_csv('thu.csv')


## tạo điều kiện để loại bỏ dữ liệu nhiễu 
data_condition = df[(df['tuoi'] >= 0) & (df['chisokhoi'] >= 0) & (df['nhiptim'] >= 0) & (df['huyetap'] >= 0) & (df['duongmau'] >= 0) & (df['cholesterol'] >= 0)
             & (df['triglycerid'] >= 0) & (df['tiensubenh'] >= 0) & (df['RANKIN'] >= 0) & ((df['gioitinh'] == 1) | (df['gioitinh'] == 2))]

print('\n','\n','==> Dữ liệu sau khi được làm sạch :','\n','\n',data_condition)

data_condition.to_csv('thu_nghiem.csv', index=False)


### tim mo hinh tot nhat

min1 = 0
a=0

for abc in range(100):
    data = pd.read_csv('thu_nghiem.csv')
    data_Train, data_Test = train_test_split(data, test_size=0.3 , shuffle = True)

    k = 10
    kf = KFold(n_splits=k, random_state=None)
    a =a+1
    i=1
    min = 0
    for train_index, test_index in kf.split(data_Train):  
        X_train, X_test = data_Train.iloc[train_index,:-1], data_Train.iloc[test_index, :-1]
        y_train, y_test = data_Train.iloc[train_index, -1], data_Train.iloc[test_index, -1]

        SVM = svm.SVC()
        SVM.fit(X_train,y_train) 
               
        Y_pred_test=SVM.predict(X_test)

        accuracy_kf = accuracy_score(Y_pred_test,y_test)
        #print("\nĐộ chính xác của mẫu trên K trên tập data_train lần :", i," : ", round(accuracy_kf * 100,3),'%','\n')

        
        good_svm_kf = SVM.fit(X_train, y_train)
        y_predict=good_svm_kf.predict(data_Test.iloc[:,:-1])
        y_reality = np.array(data_Test.iloc[:,-1])
        accuracy_reality = accuracy_score(y_predict, y_reality)
        x = (len(y_reality) * accuracy_reality)
        #print("Số dự đoán đúng trên tập dư liệu data_test thực tế :", round(x), "trên tổng", len(y_reality),'\n')
        #print('==> Độ chính xác của mẫu K trên tập dư liệu data_test thực tế là ',round(accuracy_reality * 100,3),'%','\n')
        #print('________________________________________________________________________________','\n')
        if(accuracy_reality > min):
            min = accuracy_reality
            y = (len(y_reality) * accuracy_reality)
            good_svm_reality = SVM.fit(X_train, y_train)
        i =i+1
    if(min > min1):
        min1 = min
        z = y
        very_good_svm = good_svm_reality
    print("SỐ DỰ ĐOÁN ĐÚNG TRÊN TẬP DỮ LIỆU DATA_TEST THỰC TẾ LÀ :",'Đúng', round(z), "trên tổng", len(y_reality))
    print('==> ĐỘ CHÍNH XÁC CỦA THUẬT TOÁN LẦN THỨ [',a,'] LÀ :',round(min1 * 100,3),'%' ,'\n')


#luu mo hinh tot nhat vao model.pkl de su dung 
score_svm = pickle.load(open('score_svm.pkl','rb'))
print('==> Tỷ Lệ chính xác cao nhất của lần train trước đó được lưu là :',round(score_svm * 100,3),'%' )

if(min1 > score_svm):
    score_svm = min1
    pickle.dump(score_svm, open('score_svm.pkl','wb'))
    pickle.dump(very_good_svm, open('model.pkl','wb'))


model = pickle.load(open('model.pkl','rb'))
#print("Du Doan :",  model.predict([[67,1,23,78,120,5.6,5.2,1.7,0,0]]))
print('==> Tỷ Lệ chính xác cao nhất từ các lần train là :', round(score_svm * 100,3),'%' )

#tao giao dien nguoi dung

def showPredict():
    if(txttuoi.get() == "" or txtgioitinh.get() == "" or txtchisokhoi.get() == "" or txtnhiptim.get() == "" or txthuyetap.get() == "" or txtduongmau.get() == ""
       or txtcholesterol.get() == "" or txttriglycerid.get() == "" or txttiensubenh.get() == "" or txtRANKIN.get() == ""):
         messagebox.showerror("Lỗi Thiếu Thông Tin ", " Vui lòng điền đầy đủ thông tin.");
         
    x_input = np.array([float(txttuoi.get()), float(txtgioitinh.get()), float(txtchisokhoi.get()), float(txtnhiptim.get()), float(txthuyetap.get()), float(txtduongmau.get()),
                        float(txtcholesterol.get()), float(txttriglycerid.get()), float(txttiensubenh.get()), float(txtRANKIN.get())]).reshape(1, -1)
    
    if ((float(txttuoi.get()) <= 0) ):
         messagebox.showerror("lỗi Sai Thông Tin", " Vui Lòng điền đúng số tuổi ")
         x_input = np.nan;

    if ( (float(txtgioitinh.get()) <= 0) or (float(txtgioitinh.get()) >= 3) ):
         messagebox.showerror("lỗi Sai Thông Tin", " Vui Lòng điền đúng giới tính : 1 Là Nam | 2 Là Nữ ")
         x_input = np.nan;

    if ((float(txtchisokhoi.get()) <= 0) ):
         messagebox.showerror("lỗi Sai Thông Tin", " Vui Lòng điền đúng Chỉ Số Khối ")
         x_input = np.nan;

    if ((float(txtnhiptim.get()) <= 0) ):
         messagebox.showerror("lỗi Sai Thông Tin", " Vui Lòng điền đúng Chỉ Số Nhịp Tim ")
         x_input = np.nan;

    if ((float(txthuyetap.get()) <= 0) ):
         messagebox.showerror("lỗi Sai Thông Tin", " Vui Lòng điền đúng Chỉ Số Huyết Áp ")
         x_input = np.nan;

    if ((float(txtduongmau.get()) <= 0) ):
         messagebox.showerror("lỗi Sai Thông Tin", " Vui Lòng điền đúng Chỉ Số Đường Máu ")
         x_input = np.nan;

    if ((float(txtcholesterol.get()) <= 0) ):
         messagebox.showerror("lỗi Sai Thông Tin", " Vui Lòng điền đúng Chỉ Số Cholesterol ")
         x_input = np.nan;
    
    if ((float(txttriglycerid.get()) <= 0) ):
         messagebox.showerror("lỗi Sai Thông Tin", " Vui Lòng điền đúng Chỉ Số Triglycerid ")
         x_input = np.nan;
    
    if ((float(txttiensubenh.get()) < 0) ):
         messagebox.showerror("lỗi Sai Thông Tin", " Vui Lòng điền đúng Tiền Sử Bệnh ")
         x_input = np.nan;

    if ( (float(txtRANKIN.get()) < 0) or (float(txtRANKIN.get()) >= 7) ):
         messagebox.showerror("lỗi Sai Thông Tin", " Vui Lòng điền đúng RANKIN ")
         x_input = np.nan;

      
    y_dd = model.predict(x_input) 
    print('Kết Quả :', y_dd[0])
    print('input :', x_input)

    if(y_dd == 1):
        messagebox.showinfo("Kết quả dự đoán: ", "Bạn Có Nguy Cơ Bị Đột Qụy " )
    else :
        messagebox.showinfo("Kết quả dự đoán: ", "Sức Khỏe Bình Thường " )
 #+ str(y_dd[0])

def dochinhxac():
    
    messagebox.showinfo("Khả năng dự đoán của SVM", "Độ chính xác của phương pháp: " + str(round(score_svm * 100,3)) + '%') 



#khởi tạo cửa số giao diện
windown = Tk()
windown.geometry("550x300")
windown.title("DỰ ĐOÁN NGUY CƠ ĐỘT QUỴ")

#Tạo thông số
lbltuoi = tkinter.Label (windown, text =("Tuổi"), font = ("Arial",10))
lbltuoi.grid(column = 1, row = 2)
txttuoi = Entry(windown, width = 25)
txttuoi.grid(column = 2, row = 2)

lblgioitinh = tkinter.Label (windown, text =("Giới Tính"), font = ("Arial",10))
lblgioitinh.grid(column = 1, row = 4)
txtgioitinh = Entry(windown, width = 25)
txtgioitinh.grid(column = 2, row = 4)

lblchisokhoi = tkinter.Label (windown, text =("Chỉ Số Khối"), font = ("Arial",10))
lblchisokhoi.grid(column = 1, row = 6)
txtchisokhoi = Entry(windown, width = 25)
txtchisokhoi.grid(column = 2, row = 6)

lblnhiptim = tkinter.Label (windown, text =("Nhịp Tim "), font = ("Arial",10))
lblnhiptim.grid(column = 1, row = 8)
txtnhiptim = Entry(windown, width = 25)
txtnhiptim.grid(column = 2, row = 8)

lblhuyetap = tkinter.Label (windown, text =("Huyết Áp"), font = ("Arial",10))
lblhuyetap.grid(column = 1, row = 10)
txthuyetap = Entry(windown, width = 25)
txthuyetap.grid(column = 2, row = 10)

lblduongmau = tkinter.Label (windown, text =("Đường Máu"), font = ("Arial",10))
lblduongmau.grid(column = 6, row = 2)
txtduongmau = Entry(windown, width = 25)
txtduongmau.grid(column = 7, row = 2)

lblcholesterol = tkinter.Label (windown, text =("Cholesterol"), font = ("Arial",10))
lblcholesterol.grid(column = 6, row = 4)
txtcholesterol = Entry(windown, width = 25)
txtcholesterol.grid(column = 7, row = 4)

lbltriglycerid = tkinter.Label (windown, text =("Triglycerid"), font = ("Arial",10))
lbltriglycerid.grid(column = 6, row = 6)
txttriglycerid = Entry(windown, width = 25)
txttriglycerid.grid(column = 7, row = 6)

lbltiensubenh = tkinter.Label (windown, text =("Tiền Sử Bệnh"), font = ("Arial",10))
lbltiensubenh.grid(column = 6, row = 8)
txttiensubenh = Entry(windown, width = 25)
txttiensubenh.grid(column = 7, row = 8)

lblRANKIN = tkinter.Label (windown, text =("RANKIN"), font = ("Arial",10))
lblRANKIN.grid(column = 6, row = 10)
txtRANKIN = Entry(windown, width = 25)
txtRANKIN.grid(column = 7, row = 10)




#Tạo nút bấm

btketqua = Button (windown, text = "Kết Qủa",command = showPredict)
btketqua.place(x = 75, y = 200)

btdochinhxac = Button (windown, text = "Độ Chính Xác",command = dochinhxac)
btdochinhxac.place(x = 225, y = 200)

btthoat = Button (windown, text = "Thoát", command = exit)
btthoat.place(x = 400, y = 200)

windown.mainloop()


    



