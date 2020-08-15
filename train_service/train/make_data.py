from app.service import constant , paths
import pandas as pd
from app.service import utils
from app.service.models import *
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from vncorenlp import VnCoreNLP
from pyvi import ViTokenizer
import numpy as np
from app.service import preprocessing as prep

print("Updating Data................")
# mapping 0 -> positive , 1 -> negative , 2 -> neutral
le = LabelEncoder()
le.fit(["tich_cuc" , "tieu_cuc" , "trung_tinh"])

# object tokenizer
vn_tokenizer = VnCoreNLP(paths.vncore_jar_path,
                         annotators="wseg", max_heap_size='-Xmx500m')

print("==================== Updating Train Dataset ====================")
train_df = pd.read_excel(paths.train_path)

# label encoder
y_train = train_df["label"].values
y_train = le.transform(y_train)

# tach tu cho du lieu
train_df["cau_hoi"] = train_df["cau_hoi"].progress_apply(lambda x: ' '.join([' '.join(sent) 
                                            for sent in vn_tokenizer.tokenize(x)]))

X_train = train_df["cau_hoi"].values

# tien xu ly , agumentation du lieu
X_pos = X_train[y_train == 0]
X_neg = X_train[y_train == 1]
X_neu = X_train[y_train == 2]

pos_list = [ViTokenizer.tokenize(w) for w in constant.tu_dien_tich_cuc]
neg_list = [ViTokenizer.tokenize(w) for w in constant.tu_dien_tieu_cuc]

print(X_pos)
print(pos_list)

X_positive = np.concatenate((X_pos , np.asarray(pos_list)) , axis = 0)
X_positive = prep.preprocess_corpus(X_positive)

y_positive = []
for i in range(X_positive.shape[0]):
    y_positive.append(0)

X_negative = np.concatenate((X_neg , np.asarray(neg_list)) , axis = 0)
X_negative = prep.preprocess_corpus(X_negative)

y_negative = []
for i in range(X_negative.shape[0]):
    y_negative.append(1)

X_neu = prep.preprocess_corpus(X_neu)
y_neu = []
for i in range(X_neu.shape[0]):
    y_neu.append(2)

X = np.concatenate((X_positive , X_negative , X_neu) , axis = 0)
y = np.concatenate((y_positive , y_negative , y_neu) , axis = 0)

y = y.astype('int')

X , y = shuffle(X , y)

utils.save_to_pickle_file(X , './data/X_train.pkl')
utils.save_to_pickle_file(y , './data/y_train.pkl')

print("Update Data Train Done !!!")

print("==================== Updating Test Dataset ====================")

test_df = pd.read_excel(paths.test_path)

y_test = test_df["label"].values
y_test = le.transform(y_test)

test_df["cau_hoi"] = test_df["cau_hoi"].progress_apply(lambda x: ' '.join([' '.join(sent) 
                                            for sent in vn_tokenizer.tokenize(x)]))

X_test = test_df["cau_hoi"].values
X_test = prep.preprocess_corpus_test(X_test)

utils.save_to_pickle_file(X_test , './data/X_test.pkl')
utils.save_to_pickle_file(y_test , './data/y_test.pkl')

print("Update Data Test Done !!!")