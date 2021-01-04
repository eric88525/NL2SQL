import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import *
import cn2an
import re

class N2Sm1(nn.Module):
  def __init__(self,model_type,config):
    super(N2Sm1,self).__init__()
    self.bert_model = BertModel.from_pretrained(model_type,config = config)
    self.cond_conn_op_decoder = nn.Linear(config.hidden_size,3)
    self.agg_deocder = nn.Linear(config.hidden_size,7)
    self.cond_op_decoder = nn.Linear(config.hidden_size,5)

  # 取得header_ids 所標記的 token
  def get_agg_hiddens(self,hiddens,header_ids):
    # header_ids [bsize,headers_idx]
    # hiddens [bsize,seqlength,worddim]
    arr = []
    for b_idx in range(0,hiddens.shape[0]):
      s = torch.stack([hiddens[b_idx][i] for i in header_ids[b_idx]],dim=0)
      arr.append(s)
    return torch.stack(arr,dim=0)

  def forward(self,input_ids=None,attention_mask=None,token_type_ids=None,header_ids=None): 
    # hidden [bsize,seqlength,worddim] cls [bsize,worddim]
    hiddens,cls = self.bert_model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
    cond_conn_op = self.cond_conn_op_decoder(cls)
    header_hiddens = self.get_agg_hiddens(hiddens,header_ids)
    agg = self.agg_deocder(header_hiddens)
    cond_op = self.cond_op_decoder(header_hiddens)
    # cond_conn_op [bsize,3]   
    # cond_op [bize,header_length,5]
    # agg [bize,header_length,7] 
    return cond_conn_op,cond_op , agg

class N2Sm2(nn.Module):
  def __init__(self,model_type,config):
    super(N2Sm2,self).__init__()
    self.bert_model = BertModel.from_pretrained(model_type,config = config)
    self.decoder = nn.Sequential(
        nn.Linear(config.hidden_size,1),
        nn.Sigmoid()
    )
    
  def forward(self,input_ids=None,attention_mask=None,token_type_ids=None,header_ids=None): 
    # hidden [bsize,seqlength,worddim] cls [bsize,worddim]
    hiddens,cls = self.bert_model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
    return self.decoder(cls)

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def cn_to_an(string):
    try:
        return str(cn2an.cn2an(string, 'normal'))
    except ValueError:
        return string

def an_to_cn(string):
    try:
        return str(cn2an.an2cn(string))
    except ValueError:
        return string

def str_to_num(string):
    try:
        float_val = float(cn_to_an(string))
        if int(float_val) == float_val:   
            return str(int(float_val))
        else:
            return str(float_val)
    except ValueError:
        return None

def str_to_year(string):
    year = string.replace('年', '')
    year = cn_to_an(year)
    if is_float(year) and float(year) < 1900:
        year = int(year) + 2000
        return str(year)
    else:
        return None
    
def load_json(json_file):
    result = []
    if json_file:
        with open(json_file) as file:
            for line in file:
                result.append(json.loads(line))
    return result

def extract_values_from_text(text):
        values = []
        values += extract_year_from_text(text)
        values += extract_num_from_text(text)
        return list(set(values))

def extract_num_from_text(text):
    CN_NUM = '〇一二三四五六七八九零壹贰叁肆伍陆柒捌玖貮两'
    CN_UNIT = '十拾百佰千仟万萬亿億兆点'
    values = []
    num_values = re.findall(r'[-+]?[0-9]*\.?[0-9]+', text)
    values += num_values
        
    cn_num_unit = CN_NUM +CN_UNIT
    cn_num_texts = re.findall(r'[{}]*\.?[{}]+'.format(cn_num_unit, cn_num_unit), text)
    cn_num_values = [str_to_num(text) for text in cn_num_texts]
    values += [value for value in cn_num_values if value is not None]
    
    cn_num_mix = re.findall(r'[0-9]*\.?[{}]+'.format(CN_UNIT), text)
    for word in cn_num_mix:
        num = re.findall(r'[-+]?[0-9]*\.?[0-9]+', word)
        for n in num:
            word = word.replace(n, an_to_cn(n))
        str_num = str_to_num(word)
        if str_num is not None:
            values.append(str_num)
    return values


def extract_year_from_text(text):
  values = []
  CN_NUM = '〇一二三四五六七八九零壹贰叁肆伍陆柒捌玖貮两'
  CN_UNIT = '十拾百佰千仟万萬亿億兆点'
  num_year_texts = re.findall(r'[0-9][0-9]年', text)
  values += ['20{}'.format(text[:-1]) for text in num_year_texts]
  cn_year_texts = re.findall(r'[{}][{}]年'.format(CN_NUM,CN_NUM), text)
  cn_year_values = [str_to_year(text) for text in cn_year_texts]
  values += [value for value in cn_year_values if value is not None]
  return values    

# 終極版本
class NL2SQL():
  def __init__(self,config):
    # device
    self.device = config['device']
    bert_config = BertConfig.from_pretrained(config['config_path'])
    self.tokenizer = BertTokenizer.from_pretrained(config['vocab_path'])
    self.model_1 = N2Sm1(config['model_type'],bert_config).to(self.device)
    self.model_1.load_state_dict(torch.load(config['m1_path'])) 
    self.model_2 = N2Sm2(config['model_type'],bert_config).to(self.device)
    self.model_2.load_state_dict(torch.load(config['m2_path']))
    self.analyze = config['analyze']
  def get_m1_output(self,question,headers):
    all_tokens = self.tokenizer.tokenize(question)
    col_type_token_dict = {'text': '[unused11]', 'real': '[unused12]'}
    for h,t in zip(headers[0],headers[1]):
      tokens = ['[SEP]',col_type_token_dict[t]] + self.tokenizer.tokenize(h)
      all_tokens = all_tokens + tokens
    # get the header token place 
    header_ids = []
    for i in range(len(all_tokens)):
      if all_tokens[i] == col_type_token_dict['text'] or all_tokens[i] == col_type_token_dict['real']:
        header_ids.append(i+1)
    plus = self.tokenizer.encode_plus(all_tokens)
    input_ids,token_type_ids,attention_mask,header_ids = torch.tensor(plus['input_ids']).unsqueeze(0).to(self.device)\
    ,torch.tensor(plus['token_type_ids']).unsqueeze(0).to(self.device),\
      torch.tensor(plus['attention_mask']).unsqueeze(0).to(self.device), torch.tensor(header_ids).unsqueeze(0).to(self.device)
    # pred
    cond_conn_op_pred,conds_ops_pred ,agg_pred = self.model_1(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,header_ids=header_ids)
    result = {}
    result['conn_op'] = torch.argmax(cond_conn_op_pred.squeeze(),dim=-1).to('cpu').numpy()
    result['agg'] = torch.argmax(agg_pred.squeeze(),dim=-1).to('cpu').numpy()
    result['cond'] = torch.argmax(conds_ops_pred.squeeze(),dim=-1).to('cpu').numpy()
    return result
  def get_m2_output(self,question,cond):
    ids =  self.tokenizer.encode_plus(question,cond)
    for k in ids.keys():
      ids[k] = torch.tensor(ids[k]).to(self.device)
    input_ids,token_type_ids,attention_mask = ids['input_ids'].unsqueeze(0).to(self.device),ids['token_type_ids'].unsqueeze(0).to(self.device),ids['attention_mask'].unsqueeze(0).to(self.device)
    pred = self.model_2(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids).squeeze().item()
    return pred

  def get_sql(self,data,m1,table,table_name):
   # {'agg': array([6, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]),
   # 'cond': array([4, 4, 4, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4]),
   # 'conn_op': array(1)}

    print("sql======================")
    print(table)
    print(m1)
    print("sql======================")
    conn_map = ['','AND','OR']
    agg_map = ['','AVG','MAX','MIN','COUNT','SUM','']
    cond_map = ['>','<','=','!=','']
    agg,cond,conn_op = m1['agg'],m1['cond'],conn_map[m1['conn_op']]
    headers = data['headers']

    pre = ''   
    column = ''
    condition = ''
    print(f"agg: {agg}")
    for col,val in enumerate(agg):
      if val!=6:
        column += pre+  f"{agg_map[val]}(`{headers[0][col]}`)"
        pre= ' ,'
    # where condition       
    pre = ''
    for col,val in enumerate(cond):
      if val!=4:
        if data['headers'][1][col] == 'text':
          values_list = set([r[col] for r in table]) # value from table
        elif data['headers'][1][col] == 'real':
          values_list = extract_values_from_text(data['question']) # value from question
          if self.analyze:
            print(values_list,'number from question')


        possible_cond = []

        for v in values_list:
          # format like apple > 10    
          if data['headers'][1][col]  == 'text':
            cond = f"`{headers[0][col]}` {cond_map[val]} \"{str(v)}\""
          else:
            cond = f"`{headers[0][col]}` {cond_map[val]} {str(v)}" 
          p = self.get_m2_output( data['question'],cond ) 
          possible_cond.append( [ cond , p ])

        print(possible_cond)

        possible_cond = sorted(possible_cond , key=lambda x: x[1] ,reverse=True )

        if conn_op == '':        
          condition += pre + possible_cond[0][0] + ' '
          pre = conn_op+' '
        else:
          for _cond,_p in possible_cond:
            if _p < 0.55:
              continue
            condition += pre + possible_cond[0][0] + ' '
            pre = conn_op+' '

    result = f'SELECT {column} FROM `{table_name}`';    
    if condition!='': 
      result += f" WHERE {condition}"

    return result
  def go(self,data):
    # data:
    #   table_id:
    #   question:
    #   headers:[['h1','h2','h3'],['text','real','text']] 
    #   table [['上海',1,'下雨'],[],[]...]
    
    m1 = self.get_m1_output(data["question"],data["headers"])
    result = self.get_sql(data,m1,data["table"],data["table_name"])
    return  result
