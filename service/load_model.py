from tqdm import tqdm
tqdm.pandas()
from vncorenlp import VnCoreNLP
from transformers import *
import argparse
from transformers.modeling_utils import *
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary
from app.service.test import test
from app.configs import paths
from app.configs.models import *
from app.configs.utils import *
from app.service.models import *
import joblib
import pickle

def load_svm_model():
    with open(paths.svm_path , 'rb') as f:
        svm_model = joblib.load(f)
    with open(paths.vectorizer_path , 'rb') as f:
        vectorizer = pickle.load(f)
    return vectorizer , svm_model

def load_phobert_model():

    device = torch.device("cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('--bpe-codes',
                        default=paths.bpe_codes_path,
                        required=False,
                        type=str,
                        help='path to fastBPE BPE'
                        )

    args = parser.parse_args()
    bpe = fastBPE(args)

    vn_tokenizer = VnCoreNLP(paths.vncore_jar_path,
                            annotators="wseg", max_heap_size='-Xmx500m')

    # config model
    config = RobertaConfig.from_pretrained(
        paths.config_path,
        output_hidden_states=True,
        num_labels=3
    )

    model_bert = RobertaForAIViVN.from_pretrained(paths.pretrained_path, config=config)
    # model_bert.cuda()

    # Load the dictionary
    vocab = Dictionary()
    vocab.add_from_file(paths.dict_path)

    '''
    if torch.cuda.device_count():
        print(f"Testing using {torch.cuda.device_count()} gpus")
        model_bert = nn.DataParallel(model_bert)
        tsfm = model_bert.module.roberta
    else:
        tsfm = model_bert.roberta
    '''

    model_bert = nn.DataParallel(model_bert)
    tsfm = model_bert.module.roberta

    model_bert.load_state_dict(torch.load(paths.phobert_path , map_location=device))
    
    return bpe , vn_tokenizer , model_bert , vocab