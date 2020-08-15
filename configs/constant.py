import pandas as pd
import pickle
from app.configs import paths

tu_dien_tich_cuc = pd.read_excel(paths.tu_dien_tich_cuc_path)['tu_tich_cuc'].values
tu_dien_tieu_cuc = pd.read_excel(paths.tu_dien_tieu_cuc_path)['tu_tieu_cuc'].values
tu_viet_tat = pd.read_excel(paths.tu_viet_tat_path)
stop_words = []

with open(paths.tu_dien_thay_the_path , "rb") as f:
	replace_list = pickle.load(f)

VN_CHARS_LOWER = u'ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđð'
VN_CHARS_UPPER = u'ẠẢÃÀÁÂẬẦẤẨẪĂẮẰẶẲẴÓÒỌÕỎÔỘỔỖỒỐƠỜỚỢỞỠÉÈẺẸẼÊẾỀỆỂỄÚÙỤỦŨƯỰỮỬỪỨÍÌỊỈĨÝỲỶỴỸÐĐ'
VN_CHARS = VN_CHARS_LOWER + VN_CHARS_UPPER

SPECIAL_CHARACTER = '%@$.,=+-!;/()"&^:|\n\t\''
SPECIAL_NAME = ['viettel' , 'facebook' , 'viettelpay' , 'data' , 'positive' , 'negative' , 'notpos' , 'notneg']

max_sequence_length = 128
fold = 0
epochs = 4
batch_size = 1
accumulation_steps = 5
lr = 1e-5
host="127.0.0.1"
port=3306
user="root"
passwd="hydeptrai"
