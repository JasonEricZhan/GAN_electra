import numpy as np
import pandas as pd
import os
import tokenizers
import string
import torch
import transformers
import torch.nn as nn
from torch.nn import functional as F
from tqdm.autonotebook import tqdm
import re
from torch.optim import lr_scheduler
from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import os
import matplotlib.pyplot as plt
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import nltk






class TweetModel_G(transformers.BertPreTrainedModel):
    def __init__(self,conf):
        super(TweetModel, self).__init__(conf)
        self.bert = transformers.RobertaModel.from_pretrained(config.ROBERTA_PATH, config=conf)
        print(self.bert)
        self.drop_out = nn.Dropout(0.1)
        self.l0 = nn.Linear(768 , 2)
        #self.l0_1 = nn.Linear(512 * 2, 2)
        self.selu = nn.SELU()
        #
        torch.nn.init.normal_(self.l0.weight, std=0.02)
        #torch.nn.init.normal_(self.l0_1.weight, std=0.02)
    
    def forward(self, ids, mask, token_type_ids):
        _, _,out = self.bert(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        out = torch.cat((out[-2],out[-1]), dim=-1)
        #self.lstm = nn.LSTM(embed_size, self.hidden_size, bidirectional=True, batch_first=True)
        out = self.drop_out(out)
        #out,_=self.gru(out)
        logits = self.l0(out)
        start_logits, end_logits = logits.split(1, dim=-1)
        #print(start_logits.size(), end_logits.size())
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        #print(start_logits.size(), end_logits.size())
        return start_logits, end_logits
    
    
    
    
    
    
    
    
    
class TweetModel_D(transformers.BertPreTrainedModel):
    def __init__(self,conf):
        super(TweetModel, self).__init__(conf)
        #self.bert = transformers.RobertaModel.from_pretrained(config.ROBERTA_PATH, config=conf)
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH, config=conf) # or bert 
        print(self.bert)
        self.drop_out = nn.Dropout(0.1)
        self.l0 = nn.Linear(768 , 2)
        #self.l0_1 = nn.Linear(512 * 2, 2)
        self.selu = nn.SELU()
        #
        torch.nn.init.normal_(self.l0.weight, std=0.02)
        #torch.nn.init.normal_(self.l0_1.weight, std=0.02)
    
    def forward(self, ids, mask, token_type_ids):
        _, _,out = self.bert(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        #out = torch.cat((out[-2],out[-1]), dim=-1)
        out=out[-1]
        #self.lstm = nn.LSTM(embed_size, self.hidden_size, bidirectional=True, batch_first=True)
        out = self.drop_out(out)
        #out,_=self.gru(out)
        logits = self.l0(out)
        start_logits, end_logits = logits.split(1, dim=-1)
        #print(start_logits.size(), end_logits.size())
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        #print(start_logits.size(), end_logits.size())
        return start_logits, end_logits
    
    
    
    
    
    
    
    

    
class TweetDataset:
    def __init__(self, tweet, sentiment, selected_text,pass_func=True):
        
        self.tweet = tweet
        self.sentiment = sentiment
        self.selected_text = selected_text
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN
        self.pass_func=pass_func
        self.real_max_len=float('-inf')
    
    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, item):
        
        if self.pass_func:
           data = process_data(
            self.tweet[item], 
            self.selected_text[item], 
            self.sentiment[item],
            self.tokenizer,
            self.max_len
           )
        
        else:
           

        
        #self.real_max_len=max(len(data["ids"]),self.real_max_len)
        return {
            'ids': torch.tensor(data["ids"], dtype=torch.long),
            'mask': torch.tensor(data["mask"], dtype=torch.long),
            'token_type_ids': torch.tensor(data["token_type_ids"], dtype=torch.long),
            'targets_start': torch.tensor(data["targets_start"], dtype=torch.long),
            'targets_end': torch.tensor(data["targets_end"], dtype=torch.long),
            'orig_tweet': data["orig_tweet"],
            'orig_selected': data["orig_selected"],
            'sentiment': data["sentiment"],
            'offsets': torch.tensor(data["offsets"], dtype=torch.long)
        }    
    

    
    
    

def calculate_jaccard_score(
    original_tweet, 
    target_string, 
    sentiment_val, 
    idx_start, 
    idx_end, 
    offsets,
    verbose=False):
    
    if idx_end < idx_start:
        idx_end = idx_start
    
        
    
    
    filtered_output  = ""
    for ix in range(idx_start, idx_end + 1):
        filtered_output += original_tweet[offsets[ix][0]: offsets[ix][1]]
        if (ix+1) < len(offsets) and offsets[ix][1] < offsets[ix+1][0]:
            filtered_output += " "

    if sentiment_val == "neutral" or len(original_tweet.split()) < 2:
        filtered_output = original_tweet

    jac =jaccard(target_string.strip(), filtered_output.strip())
    return jac, filtered_output
    
    
    
  
class GAN():
      def __init__(self,G,D,train_data,test_data):
          self.G=G
          self.D=D
          self.train,self.test=train_data,test_data
          
            
      def process_data(tweet, selected_text, sentiment, tokenizer, max_len):
    
          #text = nltk.word_tokenize(str(tweet))
          #s_tweet=''
          #for w,p in list(nltk.pos_tag(text)):
          #s_tweet+=" "+w+" "+p
          tweet = " " + " ".join(str(tweet).split())
          #tweet=s_tweet
          selected_text = " " + " ".join(str(selected_text).split())

          len_st = len(selected_text) - 1
          idx0 = None
          idx1 = None

          for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):
              if " " + tweet[ind: ind+len_st] == selected_text:
                 idx0 = ind
                 idx1 = ind + len_st - 1
                 break
          #print(ind,idx0,idx1)

          char_targets = [0] * len(tweet)
          if idx0 != None and idx1 != None:
             for ct in range(idx0, idx1 + 1):
                 char_targets[ct] = 1
    
          tok_tweet = tokenizer.encode(tweet)
          input_ids_orig = tok_tweet.ids
          tweet_offsets = tok_tweet.offsets
    
          target_idx = []
          for j, (offset1, offset2) in enumerate(tweet_offsets):
              if sum(char_targets[offset1: offset2]) > 0:
                  target_idx.append(j)
    
    
    
          if len(target_idx)==0:
             #print("0 t",target_idx)
             targets_start = 0
             targets_end = 0
          else:
             targets_start = target_idx[0]
             targets_end = target_idx[-1]
    
    
          #targets_start = target_idx[0]
          #targets_end = target_idx[-1]

          sentiment_id = {
              'positive': 1313,
              'negative': 2430,
              'neutral': 7974
            }
    
          input_ids = [0] + [sentiment_id[sentiment]] + [2] + [2] + input_ids_orig + [2]
          token_type_ids = [0, 0, 0, 0] + [0] * (len(input_ids_orig) + 1)
          mask = [1] * len(token_type_ids)
          tweet_offsets = [(0, 0)] * 4 + tweet_offsets + [(0, 0)]
          targets_start += 4
          targets_end += 4

          padding_length = max_len - len(input_ids)
          if padding_length > 0:
              input_ids = input_ids + ([1] * padding_length)
              mask = mask + ([0] * padding_length)
              token_type_ids = token_type_ids + ([0] * padding_length)
              tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)
      
    
    
          return {
              'ids': input_ids,
              'mask': mask,
              'token_type_ids': token_type_ids,
              'targets_start': targets_start,
              'targets_end': targets_end,
              'orig_tweet': tweet,
              'orig_selected': selected_text,
              'sentiment': sentiment,
              'offsets': tweet_offsets
          }
        
        
      
    
      def loss_fn(self.start_logits, end_logits, start_positions, end_positions):
          loss_fct = nn.CrossEntropyLoss()
          start_loss = loss_fct(start_logits, start_positions)
          end_loss = loss_fct(end_logits, end_positions)
          total_loss = (start_loss + end_loss)/2
          return total_loss
        
        
        
        
        
        
      def single_train(self,data_dict, model, device,reture_prob=True,scheduler=None,only_id=False):
          #model.train()
          losses = AverageMeter()
          jaccards = AverageMeter()
     
          d=data_dict
    
          #for bi, d in enumerate(tk0):
          ids = d["ids"]
          token_type_ids = d["token_type_ids"]
          mask = d["mask"]
          targets_start = d["targets_start"]
          targets_end = d["targets_end"]
          sentiment = d["sentiment"]
          orig_selected = d["orig_selected"]
          orig_tweet = d["orig_tweet"]
          targets_start = d["targets_start"]
          targets_end = d["targets_end"]
          offsets = d["offsets"]

          ids = ids.to(device, dtype=torch.long)
          token_type_ids = token_type_ids.to(device, dtype=torch.long)
          mask = mask.to(device, dtype=torch.long)
          targets_start = targets_start.to(device, dtype=torch.long)
          targets_end = targets_end.to(device, dtype=torch.long)

              
          outputs_start, outputs_end = model(
                  ids=ids,
                  mask=mask,
                  token_type_ids=token_type_ids,
          )
           #loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end)
           #loss.backward()
           #
          
              
          outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
          outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
          if reture_prob:
                 return outputs_start,outputs_end
                    
          jaccard_scores = []
          fileter=[]
          for px, tweet in enumerate(orig_tweet):
                  if not only_id:
                      selected_tweet = orig_selected[px]
                      tweet_sentiment = sentiment[px]
                      jaccard_score, fileter_srt = calculate_jaccard_score(
                      original_tweet=tweet,
                      target_string=selected_tweet,
                      sentiment_val=tweet_sentiment,
                      idx_start=np.argmax(outputs_start[px, :]),
                      idx_end=np.argmax(outputs_end[px, :]),
                      offsets=offsets[px]
                       )
                      fileter.append(fileter_srt)
                      jaccard_scores.append(jaccard_score)
                   else:
                      selected_tweet = orig_selected[px]
                      tweet_sentiment = sentiment[px]
                      idx_start,idx_end=np.argmax(outputs_start[px, :]),np.argmax(outputs_end[px, :])
                      #offsets[px]
                      if idx_end < idx_start:
                            idx_end = idx_start
                      filtered_output  = []
                      for ix in range(idx_start, idx_end + 1):
                           filtered_output += ids[offsets[ix][0]: offsets[ix][1]]
                           if (ix+1) < len(offsets) and offsets[ix][1] < offsets[ix+1][0]:
                               filtered_output += " "
                      
               
          jaccards.update(np.mean(jaccard_scores), ids.size(0))
          #losses.update(loss.item(), ids.size(0))
          
          return fileter
        
        
      def train(self,ITR,k,TRAINING_FILE):
          
            
          device = torch.device('cuda')  
            
          dfx = pd.read_csv(TRAINING_FILE)
          dfx = preprocess_df(dfx)
          
          #
          dfx_copy=dfx.copy()
          
          train_dataset = TweetDataset(
                                tweet=dfx.text.values,
                                sentiment=dfx.sentiment.values,
                                selected_text=dfx.selected_text.values
                                )
          train_data_loader = torch.utils.data.DataLoader(
                                   train_dataset,
                                   batch_size=32,
                                   num_workers=4
                                   )
          #can sample at there
          self.G.train()
          self.D.train()
          tk0 = tqdm(train_data_loader, total=len(train_data_loader))
          data_len=len(train_data_loader)
          criterion = nn.BCELoss() 
          d_optimizer = optim.Adam(self.D.parameters(), lr=d_learning_rate, betas=optim_betas)
          g_optimizer = optim.Adam(self.G.parameters(), lr=g_learning_rate, betas=optim_betas)
          
          idxs=np.random.choice(data_len,size=number_batch_each,replace=True)
            
            
          for t in range(epochs):
              for bi, d in enumerate(tk0):
                  #add input of G
                  if bi not in idxs:
                     continue
                
                  self.G.zero_grad()
                  g_input=self.single_train(d,self.G,device,reture_prob=False)
                  dfx_copy['text'].iloc[bi*batch_size:(bi+1)*batch_size,]=np.asarray(g_input)
                  dg = TweetDataset(
                                tweet=dfx_copy['text'].iloc[bi*batch_size:(bi+1)*batch_size,].text.values,
                                sentiment=dfx_copy['text'].iloc[bi*batch_size:(bi+1)*batch_size,].sentiment.values,
                                selected_text=dfx_copy['text'].iloc[bi*batch_size:(bi+1)*batch_size,].selected_text.values
                                )
                       
                  sp,ep=self.single_train(dg,self.D,device,reture_prob=True)  
                  logit=(sp+ep)/2
                  loss_gen=criterion(logit
                                     ,Variable(torch.ones(logit_X.size()[0]))) 
                  loss_gen=loss_gen.sum(-1)/batch_size
                  loss_gen.backward()
                  g_optimizer.step()
                  
                 """
                 train_g_dataset = TweetDataset(
                                tweet=dfx_copy.text.values,
                                sentiment=dfx_copy.sentiment.values,
                                selected_text=dfx_copy.selected_text.values
                                )
                 train_data_g_loader = torch.utils.data.DataLoader(
                                   train_g_dataset,
                                   batch_size=32,
                                   num_workers=4
                                   )
                     for i in range(k):
                 """
                  self.D.zero_grad()
                      
                      
                      
              
                      
                  #training D
                  
                        
                  sp,ep=self.single_train(d,self.D,device,reture_prob=True)
                  logit_X=(sp+ep)/2
                  sp,ep=self.single_train(dg,self.D,device,reture_prob=False)
                  logit_Z=(sp+ep)/2
                  loss_X= criterion(logit_X,Variable(torch.ones(logit_X.size()[0]))) 
                  loss_Z= criterion(logit_Z,Variable(torch.zeros(logit_Z.size()[0]))) 
                  loss_inner=(loss_X+loss_Z)/2
                  loss_inner=loss_inner.sum(-1)/batch_size
                  loss_inner.backward()
                  d_optimizer.step()
                      
              #idx=np.random.choice(data_len,replace=True)      
              #self.D()
              
              #loss_gen=loss_gen.sum(-1)/batch_size
              #loss_gen.backward()
              #optimizer_2.step()
              #here use all batch maybe can just sample to D
      
                  
      
