from app.configs import constant
import re
import numpy as np
from app.service.models import *
from vncorenlp import VnCoreNLP
from app.configs import paths

vn_tokenizer = VnCoreNLP(paths.vncore_jar_path,
                            annotators="wseg", max_heap_size='-Xmx500m')

class Preprocessing:

    def __init__(self, text_input):
        self.text_input = text_input

    def chuyen_ve_cau_khong_dau(self):
        __INTAB = [ch for ch in constant.VN_CHARS]
        __OUTTAB = "a" * 17 + "o" * 17 + "e" * 11 + "u" * 11 + "i" * 5 + "y" * 5 + "d" * 2
        __OUTTAB += "A" * 17 + "O" * 17 + "E" * 11 + "U" * 11 + "I" * 5 + "Y" * 5 + "D" * 2
        __r = re.compile("|".join(__INTAB))
        __replaces_dict = dict(zip(__INTAB, __OUTTAB))
        self.text_input = __r.sub(lambda m: __replaces_dict[m.group(0)], self.text_input)
        return self.text_input

    def xoa_cac_ky_tu_lap_cuoi_tu(self, word):
        if word in constant.SPECIAL_NAME:
            word = word.lower()
        else:
            word = re.sub(r'([A-Z])\1+', lambda m: m.group(1).upper(), word, flags=re.IGNORECASE)
        return word

    def remove_stop_word(self):
        self.text_input = " ".join([y for y in self.text_input.split() if y not in constant.stop_words])

    def chuan_hoa_cac_ky_tu_dac_biet(self): # tu dien replace list
        for k, v in constant.replace_list.items():
            k = k.split()
            split_str_input = self.text_input.split()
            for i in range(len(split_str_input)):
                if k[0] == split_str_input[i].split()[0]:
                    split_str_input[i] = v
            self.text_input = ' '.join(split_str_input)

    def thay_the_tu_viet_tat(self):
        backup_str_input = self.text_input
        change = ""
        s = self.text_input.split()
        for i, w in enumerate(constant.tu_viet_tat['tu_viet_tat'].values):
            if len(s) != 0:
                if w == s[0]:
                    change = self.text_input.replace(w, constant.tu_viet_tat['tu_da_sua_viet_tat'].values[i] + " ")
                    self.text_input = self.text_input.replace(w, constant.tu_viet_tat[
                        'tu_da_sua_viet_tat'].values[i] + " ")
                elif w == s[-1]:
                    change = self.text_input.replace(w, " " + constant.tu_viet_tat['tu_da_sua_viet_tat'].values[i])
                    self.text_input = self.text_input.replace(w, " " + constant.tu_viet_tat[
                        'tu_da_sua_viet_tat'].values[i])
                if 2 <= len(s) < 3:
                    if w == s[0] + " " + s[1]:
                        change = self.text_input.replace(w, constant.tu_viet_tat['tu_da_sua_viet_tat'].values[i] + " ")
                        self.text_input = self.text_input.replace(w,
                                                                  constant.tu_viet_tat[
                                                                      'tu_da_sua_viet_tat'].values[i] + " ")
                elif len(s) >= 3:
                    if w == s[0] + " " + s[1]:
                        change = self.text_input.replace(w, constant.tu_viet_tat['tu_da_sua_viet_tat'].values[i] + " ")
                        self.text_input = self.text_input.replace(w,
                                                                  constant.tu_viet_tat[
                                                                      'tu_da_sua_viet_tat'].values[i] + " ")
                    elif w == s[-2] + " " + s[-1]:
                        change = self.text_input.replace(w, " " + constant.tu_viet_tat['tu_da_sua_viet_tat'].values[i])
                        self.text_input = self.text_input.replace(w,
                                                                  " " + constant.tu_viet_tat[
                                                                      'tu_da_sua_viet_tat'].values[i])
                    w = " " + w + " "
                    if self.text_input.__contains__(w):
                        change = self.text_input.replace(w, " " + constant.tu_viet_tat[
                            'tu_da_sua_viet_tat'].values[i] + " ")
                        self.text_input = self.text_input.replace(w, " " + constant.tu_viet_tat[
                            'tu_da_sua_viet_tat'].values[i] + " ")
            else:
                break
        if change == "":
            change = backup_str_input
        self.text_input = change
    
    def chuan_hoa_cau_phu_dinh(self):
        texts = self.text_input.split()
        len_text = len(texts)
        for i in range(len_text):
            cp_text = texts[i]
            if cp_text in constant.tu_phu_dinh:
                numb_word = 2 if len_text - i - 1 >= 4 else len_text - i - 1
                for j in range(numb_word):
                    if ' '.join([' '.join(sent) for sent in vn_tokenizer.tokenize(str(texts[i + j + 1]))]) in constant.tu_tich_cuc:
                        texts[i] = 'tiêu_cực'
                        texts[i + j + 1] = ''
                    if ' '.join([' '.join(sent) for sent in vn_tokenizer.tokenize(str(texts[i + j + 1]))]) in constant.tu_tieu_cuc:
                        texts[i] = 'tích_cực'
                        texts[i + j + 1] = ''
            else:
                if ' '.join([' '.join(sent) for sent in vn_tokenizer.tokenize(str(cp_text))]) in constant.tu_tich_cuc:
                    if "question" not in self.text_input:
                        texts.append('tích_cực')
                elif ' '.join([' '.join(sent) for sent in vn_tokenizer.tokenize(str(cp_text))]) in constant.tu_tieu_cuc:
                    if "question" not in self.text_input:
                        texts.append('tiêu_cực')

        self.text_input = u' '.join(texts)
        self.text_input = self.text_input.replace(u'"', u' ')
        self.text_input = self.text_input.replace(u'️', u'')

    def chuan_hoa_cau_hoi(self):
        s = self.text_input.split()
        if len(s) == 0:
            None
        else:
            for word in constant.tu_de_hoi:
                if word in self.text_input and len(word.split()) >= 2:
                    self.text_input = self.text_input.replace(word, "question")
                    break
    
    def telex(self, word):
        vowel = list('ueoaiyêôơăâư')
        if len(word) > 9:
            return word
        map_sign = {'aa': 'â', 'dd': 'đ', 'ee': 'ê', 'oo': 'ô', }
        for ch in map_sign.keys():
            s = ch[0]
            if word.count(s) == 2:  # == or >= ? consider.
                word = word.replace(s, map_sign[ch], 1).replace(s, '')
        W = {'uu': 'ưu', 'uo': 'ươ', 'uaw': 'ưa', 'u': 'ư', 'o': 'ơ', 'a': 'ă', }
        if 'w' in word:
            for ch in W.keys():
                if ch in word:
                    word = word.replace(ch, W[ch])
                    word = word.replace('w', '')
                    break
        sign = list('sfxjr')
        S = {k: v for k, v in zip(vowel, 'ú é ó á í ý ế ố ớ ắ ấ ứ'.replace(' ', ''))}
        F = {k: v for k, v in zip(vowel, 'ù è ò à ì ỳ ề ồ ờ ằ ầ ừ'.replace(' ', ''))}
        R = {k: v for k, v in zip(vowel, 'ủ ẻ ỏ ả ỉ ỷ ể ổ ở ẳ ẩ ử'.replace(' ', ''))}
        X = {k: v for k, v in zip(vowel, 'ũ ẽ õ ã ĩ ỹ ễ ỗ ỡ ẵ ẫ ữ'.replace(' ', ''))}
        J = {k: v for k, v in zip(vowel, 'ụ ẹ ọ ạ ị ỵ ệ ộ ợ ặ ậ ự'.replace(' ', ''))}
        all_sign = {'s': S, 'f': F, 'r': R, 'x': X, 'j': J}
        first_sign = 'ưu ưi ao au ău ia iu oa ôi ơi ua ưa ưu âu ây ai'.split(' ')
        second_sign = 'uan oai oay uây uôi uya oan oă uê uô iu oa uâ iê yê ươ'.split(' ')
        third_sign = 'qua gia'.split(' ')
        pattern1 = '|'.join(first_sign)
        pattern2 = '|'.join(second_sign)
        pattern3 = '|'.join(third_sign)
        for s in sign:
            if s in word[1:]:
                pos3 = re.search(pattern3, word)
                if pos3:
                    pos3 = pos3.span()
                    if pos3[1] - 1 < word.rfind(s, 0):
                        sign_char = word[pos3[1] - 1]
                        word = word.replace(sign_char, all_sign[s][sign_char], 1)
                        word = word.replace(s, '')
                        return word
                pos2 = re.search(pattern2, word)
                if pos2:
                    pos2 = pos2.span()
                    if pos2[1] - 1 < word.rfind(s, 0):
                        sign_char = word[pos2[0] + 1]
                        word = word.replace(sign_char, all_sign[s][sign_char], 1)
                        if ''.join(word[:2]) != 'tr':
                            word = word[0] + word[1:].replace(s, '')
                        else:
                            word = word[:2] + word[2:].replace(s, '')
                        return word
                pos1 = re.search(pattern1, word)
                if pos1:
                    pos1 = pos1.span()
                    if pos1[1] - 1 < word.rfind(s, 0):
                        sign_char = word[pos1[0]]
                        word = word.replace(sign_char, all_sign[s][sign_char], 1)
                        word = word[0] + word[1:].replace(s, '')
                        return word
                for ch in vowel:
                    if ch in word:
                        idx = word.rfind(s, 0)
                        if word.index(ch) < idx:
                            word = word.replace(ch, all_sign[s][ch])
                            word = word[:idx] + word[idx + 1:]
                            return word
        return word

    def sua_loi_go_dau(self):
        sentence = self.text_input.split()
        for i, word in enumerate(sentence):
            if word not in constant.SPECIAL_NAME:
                sentence[i] = self.telex(word)
            else:
                sentence[i] = word
        self.text_input = ' '.join(sentence)
    
    def tach_tu(self):
        self.text_input = ' '.join([' '.join(sent) 
                                            for sent in vn_tokenizer.tokenize(str(self.text_input))])

    def preprocessing_sentence(self):
        self.text_input = self.text_input.lower()  # chuyển về chữ thường
        split_str = [x.strip(constant.SPECIAL_CHARACTER).lower() for x in
                     self.text_input.split()]  # loại bỏ các ký tự đặc biệt
        self.text_input = ' '.join(split_str)
        self.thay_the_tu_viet_tat()  # thay thế các từ sai
        self.sua_loi_go_dau()  # sửa lỗi gõ dấu bàn phím
        self.text_input = re.sub('\n', '', self.text_input)  # loại bỏ ký tự "\n"
        self.chuan_hoa_cac_ky_tu_dac_biet()  # chuẩn hoá lại một số ký tự
        split_str = self.text_input.split()
        self.text_input = [self.xoa_cac_ky_tu_lap_cuoi_tu(word) for word in split_str]  # xử lý các từ kéo dài
        self.text_input = ' '.join(self.text_input)
        self.tach_tu() # tách từ tiếng việt
        self.chuan_hoa_cau_hoi()  # chuẩn hoá câu hỏi
        self.chuan_hoa_cau_phu_dinh()  # xử lý các câu phủ định
        self.text_input = re.sub(r'\s+', ' ', self.text_input, flags=re.I)  # chuẩn hoá lại dấu cách
        self.remove_stop_word() # loại bỏ từ dừng
        return self.text_input


def preprocess_corpus_train(corpus_data):
    corpus_data_preprocessed = []
    for sentence in corpus_data:
        sentence = str(sentence)
        app = Preprocessing(sentence)
        pro_sentence = app.preprocessing_sentence()
        if pro_sentence == "":
            corpus_data_preprocessed.append("none")
        else:
            processed_sentence = app.chuyen_ve_cau_khong_dau()
            corpus_data_preprocessed.append(pro_sentence)
            corpus_data_preprocessed.append(processed_sentence)
    return np.asarray(corpus_data_preprocessed)


def preprocess_corpus_test(corpus_data):
    corpus_data_preprocessed = []
    for sentence in corpus_data:
        sentence = str(sentence)
        app = Preprocessing(sentence)
        pro_sentence = app.preprocessing_sentence()
        if pro_sentence == "":
            corpus_data_preprocessed.append("none")
        else:
            corpus_data_preprocessed.append(pro_sentence)
    return np.asarray(corpus_data_preprocessed)

