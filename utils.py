import numpy as np
import pandas as pd

register = pd.read_csv('/mnt/datasets/fusai/user_register_log.txt',sep='\t',header=None,dtype={0:np.int32,1:np.int8,2:np.int16,3:np.int16}).rename(columns={0:'user_id',1:'day',2:'register_type',3:'device_type'})
activity = pd.read_csv('/mnt/datasets/fusai/user_activity_log.txt',sep='\t',header=None,dtype={0:np.int32,1:np.int8,2:np.int8,3:np.int32,4:np.int32,5:np.int8}).rename(columns={0:'user_id',1:'day',2:'page',3:'video_id',4:'author_id',5:'action_type'})
launch = pd.read_csv('/mnt/datasets/fusai/app_launch_log.txt',sep='\t',header=None,dtype={0:np.int32,1:np.int8}).rename(columns={0:'user_id',1:'day'})
video = pd.read_csv('/mnt/datasets/fusai/video_create_log.txt',sep='\t',header=None,dtype={0:np.int32,1:np.int8}).rename(columns={0:'user_id',1:'day'})

def gen_truth(start_date,span=7):
    end_date = start_date+span
    # 保证都是已注册用户
    basic = register[register.day<start_date]
    basic = basic['user_id'].unique()
    
    u1 = launch[(launch.day>=start_date)&(launch.day<end_date)]
    u1 = u1['user_id'].unique()

    u2 = video[(video.day>=start_date)&(video.day<end_date)]
    u2 = u2['user_id'].unique()
    
    u3 = activity[(activity.day>=start_date)&(activity.day<end_date)]
    u3 = u3['user_id'].unique()
    
    truth = set(u1)|set(u2)|set(u3)
    truth = truth&set(basic)
    truth = pd.DataFrame(list(truth),columns=['user_id'])
    truth['label'] = 1
    return truth
    
def gen_label(start,end):
    max_len = end-start+1
    data = register[['user_id']].copy()
    for i in range(start,end+1):
        sub = register[register.day<=i][['user_id']].copy()
        truth = gen_truth(i+1)
        truth.columns = ['user_id','day%d_label'%i]
        sub = sub.merge(truth,'left','user_id')
        sub = sub.fillna(0)
        data = data.merge(sub,'left','user_id')
        data = data.fillna(-1)
    del data['user_id']
    data = data.values
    label_seq = []
    label_length = []
    for t in data:
        tt = list(t[t!=-1])
        l = len(tt)
        if l>0:
            label_seq.append(tt+(max_len-l)*[0])
            label_length.append(l)        
    return np.array(label_seq), label_length

def get_table(table):
    if table == 'launch':
        return launch
    elif table == 'reg':
        return register
    elif table == 'video':
        return video
    elif table == 'act':
        return activity

def gen_day_seq(start,end,table,type_columns=None,type_value=None):
    max_len = end-start+1
    data = register[['user_id']].copy()
    for i in range(start,end+1):
        sub = register[register.day<=i][['user_id']].copy()
        t = get_table(table)
        t = t[t.day==i]
        if table == 'act':
            t = t[t[type_columns]==type_value]
        t = t[['user_id']].drop_duplicates()
        t['day%d'%i] = 1
        sub = sub.merge(t,'left','user_id')
        sub = sub.fillna(0)
        data = data.merge(sub,'left','user_id')
        data = data.fillna(-1)
    del data['user_id']
    data = data.values
    seq = []
    for t in data:
        tt = list(t[t!=-1])
        l = len(tt)
        if l>0:
            seq.append(tt+(max_len-l)*[-1])
    return np.array(seq)
