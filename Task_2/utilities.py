import re
import contractions
import string
import pandas as pd
import numpy as np
import pickle
translate_table = dict((ord(char), ' ') for char in string.punctuation if not char in ['!','.','?',',']) 

slang_dict = {
    'thx':'thanks',
    'ty':'thank you',
    'eli':'explain like i am', #not eli5 since the number gets preprocessed prior to this step
    'gg':'good game',
    'gj':'good job',
    'gl':'good luck',
    'ff':'forfeit',
    'oc':'original content',
    'ama':'ask me anything',
    'aka':'also known as',
    'ikr':'i know right',
    'iirc':'if i recall correctly',
    'lol':'laughing out loud',
    'pls':'please',
    'nsfw':'not safe for work',
    'nsfl':'not safe for life',
    'omg':'oh my god',
    'omfg':'oh my fucking god',
    'lmao':'laughing my ass off',
    'lmfao':'laughing my fucking ass off',
    'af':'as fuck',
    'lpt':'life pro tip',
    'imo':'in my opinion',
    'dat':'that',
    'np':'no problem',
    'fml':'fuck my life',
    'wtf':'what the fuck',
    'wat':'what',
    'wot':'what',
    'wut':'what',
    'imho':'in my honest opinion',
    'fb':'facebook',
    'ft':'featuring',
    'vs':'versus',
    'fyi':'for your information',
    'asap':'as soon as possible',
    'g2g':'got to go',
    'ttyl':'talk to you later',
    'ftw':'for the win',
    'idc':'i don\'t care',
    'jk':'just kidding',
    'tfw':'that feeling when',
    'irl':'in real life',
    'btw':'by the way',
    'tldr':'too long did not read',
    'tl dr':'too long did not read',
    'dw':'do not worry',
    'yolo':'you only live once',
    'tmi':'too much information',
    'rip':'rest in peace'
}
for abv in slang_dict:
    contractions.add(abv,slang_dict[abv])

def cleaning(text):
    text = text.lower()
    text = re.sub(r" #821[67];", "\'", text) #apostrophe character
    text = re.sub(r" amp;", "&", text) #ampersand character
    text = re.sub(r" lt;", "<", text) #html < character
    text = re.sub(r" gt;", ">", text) #html > character
    text = re.sub(r"[:=xX]\^?[)dD\]]", "smiling emoji", text) #common positive emojis
    text = re.sub(r"[:=]\^?[|\[(]", "negative emoji", text) #common negative emojis
    text = re.sub(r"D[:=]", "negative emoji", text) #other common negative emojis
    text = re.sub(r"https?://\S+|www\.\S+", "url", text) #url links
    text = re.sub(r'/?u/\S+/?',"user", text) #reddit user references
    text = re.sub(r'/?r/\S+/?',"subreddit", text) #subreddit references
    text = re.sub(r'\d+[,\.]\d+[$€]'," money ", text) #money references 
    text = re.sub(r'\d+[$€]'," money ", text) #money references
    text = re.sub(r'[$€]\d+[,\.]?\d*[kK]?'," money ", text) #money references
    text = re.sub(r' #\d+;',"", text) #convert rest of unicode
    text = re.sub(r'\d+[,\.]?\d*'," number ", text) #convert numbers #recheck
    #expand common contractions
    text = contractions.fix(text)
    #remove most punctuaction
    text = text.translate(translate_table)
    #remove extra whitespaces
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text


import en_core_web_md
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
nlp=en_core_web_md.load(disable=['tok2vec','parser','senter','ner'])
ss = SnowballStemmer("english")
#wnl = WordNetLemmatizer()

def simplify(text, method='lem'):
    tokens=""
    
    if method=='stem':
        
        tokens = " ".join([ss.stem(tok) for tok in word_tokenize(text)])
        
    elif method=='lem':
        #sentences = tokenize_sentences(text)
        #for s in sentences:
        doc = nlp(text)
        tokens=" ".join([tok.lemma_ for tok in doc])
        #    tokens+=" ".join([wnl.lemmatize(token.text) for token in doc])
        #    tagged = pos_tag([token for token in word_tokenize(s)])
        #    tokens+=" ".join([wnl.lemmatize(token,get_wordnet_pos(pos)) for (token,pos) in tagged if not token in stop_words])
            
    
    return tokens

def full_preprocess(text,method='lem'):
    text = cleaning(text)
    return simplify(text, method=method)


def rolling_window(df, window_size,stride, field):
    res_map={}
    for user in df['User'].unique():
        user_df = df[df['User']==user]
        res_map[user]=(user_df['Label'].values[0],{})
        posts = user_df[field].values
        iteration=0
        for i in range(0,len(posts),stride):
            res_map[user][1][iteration]=' '.join((posts[i:i+window_size]))
            iteration+=1
    result_df = pd.DataFrame([(k,k1,v1,v[0]) for k,v in res_map.items() for k1,v1 in v[1].items()], columns = ['User','Window_id','Text','Label'])
    
    return result_df

def iterateWindows(df, field_name,request_index):
    users = df['User'].unique()
    posts={}
    for user in users:
        res = df.query("User==@user and Window_id==@request_index")[field_name]#df[(df['User']==user & df['Post_Nr']==request_index)]
        if len(res)>0:
            posts[user]=res.values[0]
        else:
            posts[user]=""
    return posts


class WordEmbeddingVectorizer:
    def __init__(self, model_name, embeddings_folder):
        self.model_name = model_name
        self.model_map = {"FASTTEXT_CC":"crawl-300d-2M-subword.bin","GLOVE_TT":"glove-twitter-200.bin", "GLOVE_CC":"glove_cc_300d.bin"}
        self.embeddings_folder = embeddings_folder
        if 'GLOVE' in self.model_name:
            import gensim
            self.model_type='GLOVE'
            self.embeddings = gensim.models.KeyedVectors.load_word2vec_format(self.embeddings_folder + self.model_map[self.model_name], binary=True)
            self.embeddings.add_vector('<pad>',np.zeros((self.embeddings.vector_size)))
            self.embeddings.add_vector('<unk>',np.mean(self.embeddings.vectors,axis=0, keepdims=True)[0])
        elif 'FASTTEXT' in self.model_name:
            import fasttext
            self.model_type='FASTTEXT'
            self.embeddings = fasttext.load_model(self.embeddings_folder + self.model_map[self.model_name])

    def vectorize(self,text, method='mean'):
        if self.model_type=='GLOVE':
            tokens = [self.embeddings.key_to_index.get(tok,self.embeddings.key_to_index['<unk>']) for tok in re.findall(r"\w+|[^\w\s]", text, re.UNICODE)]
            vector = self.__vectorize_aux(tokens, method=method)
            return vector
        elif self.model_type == 'FASTTEXT':
            vector =self.__vectorize_aux(re.findall(r"\w+|[^\w\s]", text, re.UNICODE),method=method)
            return vector
            
    def __vectorize_aux(self,tokens,method='mean'):
    
        vecs = []
        for token in tokens:
            vecs+=[self.embeddings[token]]
            
        if method=='mean':
            return np.mean(vecs, axis=0)
        elif method=='max':
            return np.max(vecs, axis=0)
        elif method=='min':
            return np.min(vecs, axis=0)
            


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score


def custom_cv(model, df, n_folds=5,sent =False ):
    skf = StratifiedKFold(n_splits=n_folds)
    user_label_df =df.drop_duplicates('User')
    users = user_label_df['User'].to_numpy()
    
    labels = user_label_df['Label'].to_numpy()
    
    f1_scores = []
    for train_index, test_index in skf.split(users, labels):
        train_users = [users[f] for f in train_index]
        test_users = [users[f] for f in test_index]

        train_folds = df[df['User'].isin(train_users)].copy()
        test_folds = df[df['User'].isin(test_users)].copy()
        X_train = pd.DataFrame(train_folds['Vector'].values.tolist(), index = train_folds.index)
        y_train = train_folds['Label']
        X_test = pd.DataFrame(test_folds['Vector'].values.tolist(), index = test_folds.index)
        y_test = test_folds['Label']
        if sent:
            X_train = np.c_[X_train,train_folds['polarity'],train_folds['subjectivity'],train_folds['negativity'],train_folds['positivity'],train_folds['neutrality'], train_folds['compound']] 
            X_test = np.c_[X_test,test_folds['polarity'],test_folds['subjectivity'],test_folds['negativity'],test_folds['positivity'],test_folds['neutrality'], test_folds['compound']] 
            

        
        
        model.fit(X_train, y_train)
        f1_scores.append(f1_score(y_test,model.predict(X_test)))

    return f1_scores


def sa_features(df):
    from textblob import TextBlob
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    sia = SentimentIntensityAnalyzer()
    df['TB'] = df['Text'].apply(lambda text: TextBlob(text).sentiment)
    df['VADER'] = df['Text'].apply(lambda text: sia.polarity_scores(text))


    df['polarity'] = df['TB'].apply(lambda tb: (tb[0]+1/2))
    df['subjectivity'] = df['TB'].apply(lambda tb: tb[1])
    df['negativity'] = df['VADER'].apply(lambda v: v['neg'])
    df['positivity'] = df['VADER'].apply(lambda v: v['pos'])
    df['neutrality'] = df['VADER'].apply(lambda v: v['neu'])
    df['compound'] = df['VADER'].apply(lambda v: (v['compound']+1)/2)



    df.drop(['VADER','TB'], inplace=True, axis=1)
    return df




def load_embeddings(filepath):
    with open(filepath, "rb") as fIn:
        stored_data = pickle.load(fIn)
        stored_embeddings = stored_data['embeddings']
    return stored_embeddings  


def save_embeddings(filepath, embeddings):
    with open(filepath, "wb") as fOut:
        pickle.dump({ 'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)





import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
seed=23
torch.manual_seed(seed)


class WritingWindowDataset(Dataset):
    def __init__(self, vectors, labels):

        self.labels = [label for label in labels]
        self.vectors = [vector for vector in vectors]
    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_vectors(self, idx):
        # Fetch a batch of inputs
        return self.vectors[idx]

    def __getitem__(self, idx):

        batch_vectors = self.get_batch_vectors(idx)
        batch_y = self.get_batch_labels(idx)
        

        return batch_vectors, batch_y


class LmNeuralNetwork(nn.Module):
    def __init__(self, trial):
        super(LmNeuralNetwork, self).__init__()
        self.layers=[]
        n_layers = trial.suggest_int("n_layers", 1, 3)

        in_features = 768
        for i in range(n_layers):
            out_features = trial.suggest_int("n_units_l{}".format(i), 4, min(in_features,128))
            self.layers.append(nn.Linear(in_features, out_features))
            self.layers.append(nn.ReLU())
            p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5, step=0.05)
            self.layers.append(nn.Dropout(p))

            in_features = out_features
        self.layers.append(nn.Linear(in_features, 1))
        self.cls_layers = torch.nn.ModuleList(self.layers)
    def forward(self, x):

        for layer in self.cls_layers:
            x = layer(x)
        return torch.sigmoid(x)
