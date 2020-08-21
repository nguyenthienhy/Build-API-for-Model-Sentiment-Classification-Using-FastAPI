import os
from typing import List
from fastapi import FastAPI, File, UploadFile, Request , Form ,Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
from app.service.test import test
from app.configs import constant
import uvicorn
from app.configs.utils import *
from app.service import load_model
# from app.configs import connect_database

app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# load model
bpe , vn_tokenizer , model_bert , vocab = load_model.load_phobert_model()
vectorizer , svm_model = load_model.load_svm_model()

# connect database
# curror , db = connect_database.connect_to(host=constant.host, port=constant.port, user=constant.user
                                            #, passwd=constant.passwd, db="sentiment")

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("root.html", {"request": request})

'''
@app.post("/predict_corpus/")
async def create_upload_files(request: Request , files: List[UploadFile] = File(...)):
    file_objects = [file.file for file in files]
    upload_folders = [open(os.path.join('app/data/uploads', file.filename), 'wb+') for file in files]
    print(upload_folders[0])
    pos_lists = []
    neg_lists = []
    neu_lists = []
    for i, file_object in enumerate(file_objects):
        shutil.copyfileobj(file_object, upload_folders[i])
        test_object = test.Test(x_test_file_path=upload_folders[i].name, y_test_file_path=None
                                , has_label=False
                                , model_name=None
                                , model_object = None
                                , vocab=vocab
                                , vn_tokenizer=vn_tokenizer
                                , bpe=bpe
                                , vectorizer = None)
        test_object.model_name = 'phobert'
        test_object.model_object = model_bert
        test_object.vectorizer = vectorizer
        test_object.init_test_corpus()
        num_pos , num_neg , num_neu = test_object.predict_corpus()
        pos_lists.append(num_pos)
        neg_lists.append(num_neg)
        neu_lists.append(num_neu)
        del test_object
        upload_folders[i].close()
    
    return templates.TemplateResponse("report.html", {"request": request , "status" : "done" , 
                                                    "pos_lists" : pos_lists[0] , "neg_lists" : neg_lists[0] 
                                                    , "neu_lists" : neu_lists[0]})

@app.post("/predictcorpus")
async def create_upload_files(request: Request , files: List[UploadFile] = File(...)):
    file_objects = [file.file for file in files]
    upload_folders = [open(os.path.join('app/data/uploads', file.filename), 'wb+') for file in files]
    for i, file_object in enumerate(file_objects):
        shutil.copyfileobj(file_object, upload_folders[i])
        datas = pd.read_excel(upload_folders[i])
        tu_viet_tat = datas["tu_viet_tat"].values
        tu_da_sua_viet_tat = datas["tu_da_sua_viet_tat"].values
        for i , word in enumerate(tu_viet_tat):
            #query = """insert into tu_viet_tat(tu_viet_tat , tu_da_sua_viet_tat) values('%s' , '%s')""" % (word , tu_da_sua_viet_tat[i])
            query = "select * from tu_viet_tat;"
            #query = query.encode("utf8")
            curror.execute(query)
            recored = curror.fetchall()
            print(recored)
            db.commit()
        upload_folders[i].close()
    return templates.TemplateResponse("root.html", {"request": request , "status" : "done"})
'''

@app.post("/predict")
async def predict_text_input(request: Request , text_input: str = Form(...)):
    test_object = test.Test(x_test_file_path = None, y_test_file_path = None
                                , has_label = False
                                , model_name = None
                                , model_object = None
                                , vocab = vocab
                                , vn_tokenizer = vn_tokenizer
                                , bpe = bpe
                                , vectorizer = None)
    test_object.model_name = 'phobert'
    test_object.model_object = model_bert
    #test_object.vectorizer = vectorizer
    label , text_prep = test_object.predict(text_input)
    del test_object
    return templates.TemplateResponse("root.html", {"request" : request , "text_input" : text_prep , "label" : label})


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)
