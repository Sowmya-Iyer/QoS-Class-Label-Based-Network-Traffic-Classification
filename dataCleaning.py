import glob
import pickle  
# import pyshark
# import pyasn
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import sys
from collections import Counter

import warnings
warnings.filterwarnings("ignore")

dscp_tab = {0: "Best Effort",
            8: "Priority",
            10: "Priority",
            12: "Priority",
            14: "Priority",
            16: "Immediate",
            18: "Immediate",
            20: "Immediate",
            22: "Immediate",
            24: "Flash voice",
            26: "Flash voice",
            28: "Flash voice",
            30: "Flash voice",
            32: "Flash Override",
            34: "Flash Override",
            36: "Flash Override",
            38: "Flash Override",
            40: "Critical voice RTP",
            46: "Critical voice RTP",
            48: "Internetwork control",
            56: "Network Control"
            }
dscp2v = {"CS6": "Internetwork control",
          "CS7": "Network Control",
          "CS5": "Critical voice RTP",
          "EF": "Critical voice RTP",
          "CS4": "Flash Override",
          "AF4":"Flash Override",
          "CS3":"Flash voice",
          "AF3":"Flash voice",
          "CS2":"Immediate",
          "AF2":"Immediate",
          "CS1":"Priority",
          "AF1":"Priority",
          "CS0":"Best Effort",  
          "LE": "Priority" 
}

def get_arguments(argv):
    parser = argparse.ArgumentParser(description='Creating data files for training')
    parser.add_argument('data_folder', metavar='DATA_FOLDER',
                        help='the folder that contains all the input data')
    args = parser.parse_args(argv)
    return args

def string2numeric(x):
  if "str" in str(type(x)):
    if "," in x:
      return x.split(",")[0]
  return x


def create_data(data_dir):
    # data_dir=args.data_folder
    list_files = glob.glob(data_dir+"*.csv")
    for ex in list_files:
        try:  
            os.mkdir('dump')
        except OSError:  
            print ("Folder exists")
        else:
            print ("Created the directory %s" % 'dump')
        df= pd.read_csv(ex, header = 0, index_col=0)
        df.rename(columns={"Total Length":"length","Time to Live":"ttl","Source":"ip_src","Destination":"ip_dst","Fragment Offset":"fragment_offset","More fragments":"flag_mf","Don't fragment": "flag_df","Reserved bit":"flag_rb"},inplace=True)
        required_columns=required_columns=['Protocol','flag_df','flag_mf','flag_rb','IHL','length','ttl','ip_src','ip_dst','fragment_offset','dscp v','dscp']
        for col in df.columns:
            if not col in required_columns:
                df= df.drop(col, axis=1)
        df["dscp v"] =df["dscp v"].apply(string2numeric)
    # df["dscp v"] =df["dscp v"].fillna(-1)
    # df["dscp"] =df["dscp"].fillna(-1)
        df["Label"]=np.nan
    # df["dscp v"]= pd.to_numeric(df['dscp v'], errors='coerce')
        for i in range(1,df.shape[0]):
      # print(df.loc[i,"dscp"])
            if ' ' in str(df.loc[i, 'dscp']):
                df.loc[i, 'dscp']= str(df.loc[i, 'dscp']).split(' ')[0]
            if not pd.isna(df.loc[i,"dscp v"]):
                if int(df.loc[i,"dscp v"]) in dscp_tab:
                    df.loc[i,"Label"] = dscp_tab[int(df.loc[i,"dscp v"])]
                    continue
            if str(df.loc[i, 'dscp']) in dscp2v:
                df.loc[i,"Label"] = dscp2v[str(df.loc[i, 'dscp'])]
                continue
            df.loc[i,"Label"] = "Not Known"
    # column_drop=['Type','Code','Differentiated Services Field','Length','dscp','dscp v','ds_field','ds_field_ecn',"Differentiated Services Codepoint",]
    # for col in column_drop:
    #   if col in df.columns:
    #     df= df.drop(col, axis=1)
        # print(df.columns)
        # print(df['Protocol'].value_counts())
        df.dropna(inplace=True)
        # print(df['Protocol'].value_counts())
    # categorical=['ip_src','ip_dst','Protocol','IHL','fragment_offset','Label']
    # for cat in categorical:
    #   df[cat]=df[cat].astype('category')

    #VM:
        name=ex.split('.')[0].split('/')[-1]
    #Local:
        # name=ex.split('.')[0].split('\\')[-1]
        with open('dump/' + name + '.pkl', 'wb') as f:
            pickle.dump(df, f)

def main():
    create_data(args.data_folder)

if __name__ == '__main__':
    args = get_arguments(sys.argv[1:])
    # args = utils.bin_config(get_arguments)
    main()