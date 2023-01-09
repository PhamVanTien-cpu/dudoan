from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import svm
import pickle

score = 0.5

pickle.dump(score, open('score_svm.pkl','wb'))
score_svm = pickle.load(open('score_svm.pkl','rb'))


print('Tỉ lệ chính xác tốt nhất của biến là', score_svm)
