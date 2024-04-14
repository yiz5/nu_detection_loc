import numpy as np
import os
import pandas as pd

def flatten(l):
    return [item for sublist in l for item in sublist]

def find_max_lessthan(value, arr):
    return max(arr[arr<value])

def get_int(ele):
    if ele.is_integer():
        return int(ele)
    else:
        return ele
    
def qua2arr(datas):
    return [data.value for data in datas]*datas[0].unit

def compare_equal(data1, data2):
    check_equal = [round(d1/d2, 7)==1 for d1, d2 in zip(data1, data2) if not np.isnan(d1/d2) ]
    return all(check_equal)


def read_file_dataline(file_name):
    #print(file_name)
    f = open(file_name)
    lines = []
    for line in f:
        line_split = line.split()
        if len(line_split)>0:#check sth in a line
            #print((line_split[0]))
            try: #check data line
                data_line = float(line_split[0])
                #print(len([float(ele.replace(',', '')) for ele in line_split]))
                lines.append([float(ele.replace(',', '')) for ele in line_split])

            except ValueError as e:  #check title line
                data_line = np.array([str(i) for i in line_split])
                #print(line)
                #lines.append(data_line)

    f.close()
    #print(lines)
    return np.array(lines).T


def read_file_textline(file_name):
    #print(file_name)
    f = open(file_name)
    lines = []
    for line in f:
        line_split = line.split()
        if len(line_split)>0:#check sth in a line
            #print((line_split[0]))
            try: #check data line
                data_line = float(line_split[0])
               
            except ValueError as e:  #check title line
                data_line = line
                #print(line)
                lines.append(data_line)

    f.close()
    #print(lines)
    return np.array(lines)


def read_file_certain_pos(file_name, row_num, col_num):
    print(file_name)
    f = open(file_name)
    lines = []
    for l,line in enumerate(f):
        line_split = line.split()
        if l == row_num:
            print(line)
            print(line_split[col_num])
            return line_split[col_num]
    f.close()
    
def write_file_data(file, data,sigfig = 10, delimiter = "\t\t", append= False):
    if append:
        f_out = open(file , 'a')
    else:
        f_out = open(file , 'w')
    for d in data:
        
        if type(d) in [np.ndarray, list]:
            f_out.write((delimiter.join(['{:.{}E}' .format(a,sigfig) for a in d]))+'\n' )
        else:
            f_out.write(str(d)+'\n' )
        #f_out.write(("   ".join(["{:.7E}".format(a) for a in d]))+'\n' )
    f_out.close()        
    return

def write_file_textline(file, textlines, append= False):
    if append:
        f_out = open(file , 'a')
    else:
        f_out = open(file , 'w')
    for textline in textlines:
        
        f_out.write(textline+'\n' )
        #f_out.write(("   ".join(["{:.7E}".format(a) for a in d]))+'\n' )
    f_out.close()        
    return

def read_file_data(file):
    f_out = open(file , 'r')
    data = []
    for line in f_out:
            dataline = np.array([float(i) for i in line.split()])
            data.append(dataline)
    f_out.close()
    data = np.array(data)
    return data.T

def read_file(file):
    f_out = open(file , 'r')
    data = []
    for line in f_out:
            data.append(line.split())
    f_out.close()
  
    return data


def get_param_file_df(file_name):
    f = open(file_name)
    lines = []
    for l,line in enumerate(f):
        line_split = line.split()

        if len(line_split)>0:
            if line_split[-1]=='yr)':
                lines.append(np.array(line_split)[:-4])
            else:
                column_name = np.array(line_split)
    f.close()
    DATA =pd.DataFrame((np.array(lines).T[1::]).T, index = (np.array(lines).T[0]),columns = column_name[1:])
    return DATA