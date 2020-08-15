from app.configs import paths
import pandas as pd
import numpy as np

def is_excel(file):
    if file.split('/')[-1].__contains__('.xlsx'):
        return True
    return False

def is_text(file):
    if file.split('/')[-1].__contains__('.txt'):
        return True
    return False

def read_text(file):
    f = open(file , 'r' , encoding="utf-8-sig")
    content = f.readlines()
    return [word.replace('\n' , '') for word in content]

def read_excel(file):
    f = pd.read_excel(file)
    return f

class Update:
    def __init__(self , update_file_train = False , update_file_test = False , update_constant = False):
        self.train_file = None
        self.test_file = None
    def update_train_data(self , file_list_path , name_field_one , name_field_two):
        for file in file_list_path:
            if is_excel(file):
                add_data = read_excel(file)
                old_train = pd.read_excel(paths.train_path)
                x_new_data = np.concatenate((old_train[name_field_one].values , add_data[name_field_one].values) , axis = 0)
                y_new_data = np.concatenate((old_train[name_field_two].values , add_data[name_field_two]) , axis = 0)
                df = pd.DataFrame()
                df[name_field_one] = x_new_data
                df[name_field_two] = y_new_data
                df.to_excel(paths.train_path)
            else:
                print("File is not match format")

    def update_test_data(self , file_list_path , name_field_one , name_field_two):
        for file in file_list_path:
            if is_excel(file):
                add_data = read_excel(file)
                old_data = pd.read_excel(paths.test_path)
                x_new_data = np.concatenate((old_data[name_field_one].values , add_data[name_field_one].values) , axis = 0)
                y_new_data = np.concatenate((old_data[name_field_two].values , add_data[name_field_two]) , axis = 0)
                df = pd.DataFrame()
                df[name_field_one] = x_new_data
                df[name_field_two] = y_new_data
                df.to_excel(paths.test_path)
            else:
                print("File is not match format")

    def update_tu_viet_tat(self , file_list_path , name_field_one , name_field_two):
        for file in file_list_path:
            if is_excel(file):
                add_data = read_excel(file)
                old_data = pd.read_excel(paths.tu_viet_tat_path)
                x_new_data = np.concatenate((old_data[name_field_one].values , add_data[name_field_one].values) , axis = 0)
                y_new_data = np.concatenate((old_data[name_field_two].values , add_data[name_field_two]) , axis = 0)
                df = pd.DataFrame()
                df[name_field_one] = x_new_data
                df[name_field_two] = y_new_data
                df.to_excel(paths.tu_viet_tat_path)
            else:
                print("File is not match format")

    def update_tu_tich_cuc(self , file_list_path , name_field_one):
        for file in file_list_path:
            if is_excel(file):
                add_data = read_excel(file)
                old_data = pd.read_excel(paths.tu_dien_tich_cuc_path)
                x_new_data = np.concatenate((old_data[name_field_one].values , add_data[name_field_one].values) , axis = 0)
                df = pd.DataFrame()
                df[name_field_one] = x_new_data
                df.to_excel(paths.tu_dien_tich_cuc_path)
            else:
                print("File is not match format")

    def update_tu_tieu_cuc(self , file_list_path , name_field_one):
        for file in file_list_path:
            if is_excel(file):
                add_data = read_excel(file)
                old_data = pd.read_excel(paths.tu_dien_tieu_cuc_path)
                x_new_data = np.concatenate((old_data[name_field_one].values , add_data[name_field_one].values) , axis = 0)
                df = pd.DataFrame()
                df[name_field_one] = x_new_data
                df.to_excel(paths.tu_dien_tieu_cuc_path)
            else:
                print("File is not match format")