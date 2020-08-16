from app.configs.utils import *
from app.configs import constant , paths
from app.service import preprocessing as prep
import torch.utils.data
from torch.nn.functional import softmax
from sklearn.metrics import f1_score

class Test:
  def __init__(self , x_test_file_path = None ,
               y_test_file_path = None
               , has_label = False
               , model_name = None , model_object = None , vocab = None 
               , vn_tokenizer = None , bpe = None , vectorizer = None ):
    self.x_test_file_path = x_test_file_path
    self.y_test_file_path = y_test_file_path
    self.has_label = has_label
    self.y_test = None
    self.X_test = None
    self.model_name = model_name
    self.model_object = model_object
    self.vocab = vocab
    self.vn_tokenizer = vn_tokenizer
    self.bpe = bpe
    self.vectorizer = vectorizer
    
  def init_test_corpus(self):
    if self.has_label is True: # kiểm tra độ chính xác của mô hình trên dữ liệu dạng pickle
      self.y_test = load_pickle_file(self.y_test_file_path)
      self.X_test = load_pickle_file(self.x_test_file_path)
    else:
      test_df = pd.read_excel(self.x_test_file_path)
      test_df["cau_hoi"] = test_df["cau_hoi"].progress_apply(lambda x: ' '.join([' '.join(sent) 
                                            for sent in self.vn_tokenizer.tokenize(str(x))]))
      self.X_test = test_df["cau_hoi"].values
      self.X_test = prep.preprocess_corpus_test(self.X_test) # tiền xử lý dữ liệu
    if self.model_name == "phobert":
      self.X_test = convert_lines(self.X_test, self.vocab, self.bpe , constant.max_sequence_length)
  
  def predict_corpus(self): # cho vào 1 file , nhận diện
    count_pos = 0
    count_neg = 0
    count_neu = 0
    if self.model_name == "phobert":
      test_dataset = torch.utils.data.TensorDataset(torch.tensor(self.X_test,dtype=torch.long))
      test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=constant.batch_size, shuffle=False)
      pbar = tqdm(enumerate(test_loader),total=len(test_loader),leave=False)
      all_preds = []
      for i, (x_batch,) in pbar:
        logits = self.model_object(x_batch, attention_mask=(x_batch > 0))
        predictions = torch.argmax(softmax(logits, 1), 1)
        self.model_object.eval()
        all_preds.extend(predictions.cpu())
      all_preds = np.array(all_preds)
      export_preds = []
      for i in range(all_preds.shape[0]):
        export_preds.append("")
        if all_preds[i] == 0:
          export_preds[i] = 'tich_cuc'
          count_pos += 1
        elif all_preds[i] == 1:
          export_preds[i] = 'tieu_cuc'
          count_neg += 1
        else:
          export_preds[i] = 'trung_tinh'
          count_neu += 1
      df = pd.DataFrame()
      df["cau_hoi"] = pd.read_excel(self.x_test_file_path)["cau_hoi"].values
      df["label"] = all_preds
      df.to_excel(paths.report_path + 'report_' + str(self.x_test_file_path.split('/')[-1]))
      print("Done")
      return count_pos , count_neg , count_neu
    else:
      self.X_test = self.vectorizer.transform(self.X_test)
      all_preds = self.model_object.predict(self.X_test)
      export_preds = []
      for i in range(all_preds.shape[0]):
        export_preds.append("")
        if all_preds[i] == 0:
          export_preds[i] = 'tich_cuc'
          count_pos += 1
        elif all_preds[i] == 1:
          export_preds[i] = 'tieu_cuc'
          count_neg += 1
        else:
          export_preds[i] = 'trung_tinh'
          count_neu += 1
      df = pd.DataFrame()
      df["cau_hoi"] = pd.read_excel(self.x_test_file_path)["cau_hoi"].values
      df["label"] = all_preds
      df.to_excel(paths.report_path + 'report_' + str(self.x_test_file_path.split('/')[-1]))
      print("Done")
      return count_pos , count_neg , count_neu

  
  def test_corpus(self): # kiểm tra độ chính xác của mô hình
    if self.model_name == "phobert":
      test_dataset = torch.utils.data.TensorDataset(torch.tensor(self.X_test,dtype=torch.long))
      test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=constant.batch_size, shuffle=False)
      pbar = tqdm(enumerate(test_loader),total=len(test_loader),leave=False)
      all_preds = []
      for i, (x_batch,) in pbar:
        logits = self.model_object(x_batch, attention_mask=(x_batch > 0))
        predictions = torch.argmax(softmax(logits, 1), 1)
        self.model_object.eval()
        all_preds.extend(predictions.cpu())
      all_preds = np.array(all_preds)
      score = f1_score(self.y_test, np.array(all_preds), average='macro')
      print("F1 score : " + str(score))
    else:
      self.X_test = self.vectorizer.transform(self.X_test)
      all_preds = self.model_object.predict(self.X_test)
      score = f1_score(self.y_test, np.array(all_preds), average='macro')
      print("F1 score : " + str(score))
  
  def predict(self , text_input):
    app = prep.Preprocessing(text_input)
    text_preped = app.preprocessing_sentence()
    if self.model_name == "phobert":
      X_test = convert_lines([text_preped], self.vocab, self.bpe,constant.max_sequence_length)
      text_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test,dtype=torch.long))
      test_loader = torch.utils.data.DataLoader(text_dataset, batch_size = 1)
      for (x_batch,) in test_loader:
        logits = self.model_object(x_batch, attention_mask=(x_batch > 0))
        predictions = torch.argmax(softmax(logits, 1), 1)
        self.model_object.eval()
        if predictions.cpu().item() == 0:
          return "Tích cực" , text_preped
        elif predictions.cpu().item() == 1:
          return "Tiêu cực" , text_preped
        else:
          return "Trung tính" , text_preped
    else:
      X_test = self.vectorizer.transform([text_preped])
      label_pred = self.model_object.predict(X_test)
      if label_pred == 0:
        return "Tích cực" , text_preped
      elif label_pred == 1:
        return "Tiêu cực" , text_preped
      else:
        return "Trung tính" , text_preped

