import numpy as np
import pandas as pd
import socket, struct
from dataCleaning import string2numeric
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from collections import Counter
from sklearn.utils import shuffle

def ip2int(ip):
    if "str" in str(type(ip)):
      packedIP = socket.inet_aton(ip)
      return struct.unpack("!L", packedIP)[0]
    else:
      return ip

def preprocess(df,label_col, label=True):
    print("Preprocessing data...")
    df.rename(columns={"Total Length":"length","Time to Live":"ttl","Source":"ip_src","Destination":"ip_dst","Fragment Offset":"fragment_offset","More fragments":"flag_mf","Don't fragment": "flag_df","Reserved bit":"flag_rb"},inplace=True)
    categorical=['Protocol']
    numeric=['flag_df','flag_mf','flag_rb','IHL']
    scale=['length','ttl']
    for col in numeric+scale:
      df[col] = df[col].apply(string2numeric)
      if col in ['flag_df','flag_mf','flag_rb']:
        df[col] =df[col].replace({"Not set": int(0), "Set": int(1)})
      df[col]= pd.to_numeric(df[col])
    # df["flag_df"] =df["flag_df"].apply(string2numeric)
    # df["flag_mf"] =df["flag_mf"].apply(string2numeric)
    # df['length'] =df['length'].apply(string2numeric)
    # df["flag_rb"] =df["flag_rb"].apply(string2numeric)
    # df['ttl']=df['ttl'].apply(string2numeric)
    # df['flag_df']=df['flag_df'].replace({"Not set": int(0), "Set": int(1)})
    # df['flag_mf']=df['flag_mf'].replace({"Not set": int(0), "Set": int(1)})
    # df["flag_rb"] =df["flag_rb"].replace({"Not set": int(0), "Set": int(1)})
    # numeric=['flag_df','flag_mf','length','ttl','flag_rb']
    # for col in numeric:
    #   df[col]= pd.to_numeric(df[col])
    mms = MinMaxScaler()
    mms.fit(df[scale])
    data_transformed = mms.transform(df[scale])
    data_transformed = pd.DataFrame(data_transformed, columns= scale)
    data_transformed.dropna(inplace=True)
    data_transformed[numeric]= df[numeric]
    data_transformed[categorical] = df[categorical]
    # data_transformed=data_transformed.loc[data_transformed['Protocol'].isin(['TCP','ICMP'])]
    for col in categorical:
            dummies = pd.get_dummies(data_transformed[col], prefix=col)
            data_transformed = pd.concat([data_transformed, dummies], axis=1)
            data_transformed.drop(col, axis=1, inplace=True)
    # if not 'Protocol_ICMP' in data_transformed.columns:
    #   data_transformed['Protocol_ICMP'] =0
    # elif not 'Protocol_TCP' in data_transformed.columns:
    #   data_transformed['Protocol_TCP'] =0
    remaining=['ip_src','ip_dst','fragment_offset']
    data_transformed[remaining] = df[remaining]
    # for col in remaining:
    #   data_transformed[col]=data_transformed[col].astype('category') 
    data_transformed['ip_src'] = data_transformed['ip_src'].apply(ip2int)
    data_transformed['ip_dst'] = data_transformed['ip_dst'].apply(ip2int)
    # data_transformed['IHL']=data_transformed['IHL'].apply(string2numeric)
    # data_transformed['IHL']=data_transformed['IHL'].astype('float')
    data_transformed['fragment_offset']=data_transformed['fragment_offset'].apply(string2numeric)
    data_transformed['fragment_offset']=data_transformed['fragment_offset'].astype('float')
    required_columns=['flag_df','flag_mf','flag_rb','IHL','length','ttl','ip_src','ip_dst','fragment_offset']
    prot_dummies=['Protocol_TCP','Protocol_UDP','Protocol_ICMP','Protocol_DNS','Protocol_ESP','Protocol_IPv4','Protocol_ARP','Protocol_ICMPv6','Protocol_ETHERIP']
    for prot in prot_dummies:
      if prot not in data_transformed.columns:
        data_transformed[prot]=0
    for col in data_transformed.columns:
        if col not in prot_dummies+required_columns:
          data_transformed=data_transformed.drop(col,axis=1)
    if label:
      data_transformed[label_col]=df[label_col]
      data_transformed=data_transformed[required_columns+prot_dummies+["Label"]]
    else:
      data_transformed=data_transformed[required_columns+prot_dummies]
    data_transformed.dropna(inplace=True)
    data_transformed=shuffle(data_transformed)
    data_transformed.reset_index(drop=True)
    print("Data has %s entries"% data_transformed.shape[0])
    return data_transformed