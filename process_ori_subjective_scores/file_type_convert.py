
from tqdm import tqdm
import json
import pandas as pd
import matplotlib.pyplot as plt
omit=['Test User','Ouyang Fei', 'Xiao Xinyao', 'Tian Menghan', 'Lv Xiao', 'Zeng Zhirui', 'Xie Chunlong', 'Li Weiyang', 'Qin Zhen', 'Pan Hongyang', 'Ran Tao', 'Zhang Shubo', 'Ouyang Zhaobin', 'Huanghui']
ME_10=['Wang Hao', 'Lv Xiao',
       'Li Hao', 'He Wenjian', 'Wang Yu','Wang Zhongming', 'Li Weiyang', 'Xie Chunlong']
RESULT = "./result.json"

PARAMETERs={'00':[0,1,2,3,4],
            '01':[0,1,2,3,4],
            '02':[0,1,2,3,4],
            '03':[0,1,2,3,4],
            '04':[0,1,2],
            '05':[0,],
            '06':[0,1,2],
            '07':[0,1],
            '08':[0,1],
            '09':[0,1,2,3,4,5],#,3,4,5
            '10':[0,],
            '11':[0,]}

# 将原始记录result.json转为处理后的平均分
def convert_ave_from_original_core(source: dict)->dict:
    count=0
    ave={'ID':[],
         'TM':[],
         'ME':[],
         'PA':[],
         'AE':[],
         'SS':[]}
    for id in tqdm(source.keys()):
        if int(id)<400:
            for pic in source[id]:
                if pic['AE']=='T':
                    count+=1
                    TM=pic['TM']
                    ME=pic['ME']
                    PA=pic['PA']
                    pic_name=f'{id}_{TM}_{ME}_{PA}_T'
                    ss=[]
                    if pic['ME']=='10' and count<=5782:
                        for os in pic['OS']:
                            if os['username'] in ME_10:
                                ss.append(os['os'])      
                    else:                        
                        for os in pic['OS']:
                            if os['username'] in omit:
                                continue
                            else:
                                ss.append(os['os'])
                    if len(ss)>4:
                        ss.remove(min(ss))
                        ss.remove(max(ss))

                    l=len(ss)
                    if l==0:
                        print(pic_name)
                    ss=sum(ss)/l
                    ave['ID'].append(id)
                    ave['TM'].append(TM)
                    ave['ME'].append(ME)
                    ave['PA'].append(PA)
                    ave['AE'].append('T')
                    ave['SS'].append(ss)         
    return ave

def load_json(result):
    with open(result, "r") as fp:
        dataset = json.load(fp)
    return dataset

# def convert_ave_from_original():
#     dataset=load_json(RESULT)
#     ave=convert_ave_from_original_core(dataset)
#     with open('./result_ave.json','w') as f:
#         f.write(json.dumps(ave))

def convert_csv():
    dataset=load_json(RESULT)
    dataset=convert_ave_from_original_core(dataset)
    df=pd.DataFrame(dataset)
    df=df.round(2)
    df.to_csv('./result_ave.csv',index=True)

def exception_value(g1_ME,g2_ME,g3_ME,g4_ME,box):
    outliers = [y.get_ydata() for y in box["caps"]]
    up1,low1=outliers[1][0],outliers[0][0]
    up2,low2=outliers[3][0],outliers[2][0]
    up3,low3=outliers[5][0],outliers[4][0]
    up4,low4=outliers[7][0],outliers[6][0]
    # print(up1,low1)
    g1_ME_exc=g1_ME.loc[(g1_ME['SS']>up1) | (g1_ME['SS']<low1)]
    g2_ME_exc=g2_ME.loc[(g2_ME['SS']>up2) | (g2_ME['SS']<low2)]
    g3_ME_exc=g3_ME.loc[(g3_ME['SS']>up3) | (g3_ME['SS']<low3)]
    g4_ME_exc=g4_ME.loc[(g4_ME['SS']>up4) | (g4_ME['SS']<low4)]
    return [g1_ME_exc,g2_ME_exc,g3_ME_exc,g4_ME_exc]

def revise(group,shift,ME,PA):
    if shift<0:
        group.loc[(group['ME']==int(ME)) & (group['PA']==int(PA)) & (group['SS']<80),'SS']=group.loc[(group['ME']==int(ME)) & (group['PA']==int(PA)) & (group['SS']<80)]['SS']+shift
    else:
        group.loc[(group['ME']==int(ME)) & (group['PA']==int(PA)) & (group['SS']<(80-shift-1)),'SS']=group.loc[(group['ME']==int(ME)) & (group['PA']==int(PA)) & (group['SS']<(80-shift-1))]['SS']+shift
    
def do_group(group1,group2,group3,group4):
    df={'Group':[1,2,3,4],
        'MIN':[],
        'MAX':[],
        'MEAN':[],
        'VAR':[],
        'MODE':[],
        'MEDIAN':[]}
    exception_cases=[]
    for ME in PARAMETERs.keys():
        for PA in PARAMETERs[ME]:

            # 不得改变80分，不能低于0或超过80，统一分布长度是最好的
            # if int(ME)==0 and PA==0:
            #     revise(group2,-5,ME,PA)
            if int(ME)==0 and PA==1:
                revise(group4,-5,ME,PA)
                # group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA)),'SS']=group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA))]['SS']-5
            if int(ME)==0 and PA==2:
                revise(group4,-5,ME,PA)
                # group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA)),'SS']=group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA))]['SS']-5
            if int(ME)==0 and PA==3:
                # revise(group1,-2,ME,PA)
                revise(group4,-5,ME,PA)
                # group1.loc[(group1['ME']==int(ME)) & (group1['PA']==int(PA)),'SS']=group1.loc[(group1['ME']==int(ME)) & (group1['PA']==int(PA))]['SS']-2
                # group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA)),'SS']=group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA))]['SS']-6
            if int(ME)==0 and PA==4:
                revise(group1,-10,ME,PA)
                revise(group2,-2,ME,PA)
                revise(group3,-2,ME,PA)
                revise(group4,-13,ME,PA)
                # group1.loc[(group1['ME']==int(ME)) & (group1['PA']==int(PA)),'SS']=group1.loc[(group1['ME']==int(ME)) & (group1['PA']==int(PA))]['SS']-10
                # group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA)),'SS']=group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA))]['SS']-10
            if int(ME)==1 and PA==0:
                revise(group2,+5,ME,PA)
                revise(group4,-5,ME,PA)
                # group2.loc[(group2['ME']==int(ME)) & (group2['PA']==int(PA)),'SS']=group2.loc[(group2['ME']==int(ME)) & (group2['PA']==int(PA))]['SS']+3
                # group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA)),'SS']=group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA))]['SS']-3
            if int(ME)==1 and PA==1:
                revise(group2,+5,ME,PA)
                revise(group4,-5,ME,PA)
                # group2.loc[(group2['ME']==int(ME)) & (group2['PA']==int(PA)),'SS']=group2.loc[(group2['ME']==int(ME)) & (group2['PA']==int(PA))]['SS']+3
                # group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA)),'SS']=group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA))]['SS']-3
            if int(ME)==1 and PA==2:
                revise(group4,-5,ME,PA)
                # group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA)),'SS']=group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA))]['SS']-6
            if int(ME)==1 and PA==3:
                revise(group1,-5,ME,PA)
                revise(group4,-10,ME,PA)
                # group1.loc[(group1['ME']==int(ME)) & (group1['PA']==int(PA)),'SS']=group1.loc[(group1['ME']==int(ME)) & (group1['PA']==int(PA))]['SS']-6
                # group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA)),'SS']=group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA))]['SS']-8
            if int(ME)==1 and PA==4:
                revise(group1,-5,ME,PA)
                revise(group4,-10,ME,PA)
                # group1.loc[(group1['ME']==int(ME)) & (group1['PA']==int(PA)),'SS']=group1.loc[(group1['ME']==int(ME)) & (group1['PA']==int(PA))]['SS']-6
                # group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA)),'SS']=group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA))]['SS']-8
            if int(ME)==2 and PA==0:
                revise(group2,+5,ME,PA)
                # revise(group2,+5,ME,PA)
                # revise(group3,-2.5,ME,PA)
                # group2.loc[(group2['ME']==int(ME)) & (group2['PA']==int(PA)),'SS']=group2.loc[(group2['ME']==int(ME)) & (group2['PA']==int(PA))]['SS']+3
                # group3.loc[(group3['ME']==int(ME)) & (group3['PA']==int(PA)),'SS']=group3.loc[(group3['ME']==int(ME)) & (group3['PA']==int(PA))]['SS']-2.5
            if int(ME)==2 and PA==1:
                # revise(group1,+2,ME,PA)
                revise(group2,+5,ME,PA)
                # group1.loc[(group1['ME']==int(ME)) & (group1['PA']==int(PA)),'SS']=group1.loc[(group1['ME']==int(ME)) & (group1['PA']==int(PA))]['SS']+2
                # group2.loc[(group2['ME']==int(ME)) & (group2['PA']==int(PA)),'SS']=group2.loc[(group2['ME']==int(ME)) & (group2['PA']==int(PA))]['SS']+5
            if int(ME)==2 and PA==2:
                revise(group2,+5,ME,PA)
                revise(group4,-5,ME,PA)
                # group2.loc[(group2['ME']==int(ME)) & (group2['PA']==int(PA)),'SS']=group2.loc[(group2['ME']==int(ME)) & (group2['PA']==int(PA))]['SS']+5
                # group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA)),'SS']=group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA))]['SS']-2
            if int(ME)==2 and PA==3:
                revise(group1,-5,ME,PA)
                revise(group4,-5,ME,PA)
                # group1.loc[(group1['ME']==int(ME)) & (group1['PA']==int(PA)),'SS']=group1.loc[(group1['ME']==int(ME)) & (group1['PA']==int(PA))]['SS']-4
                # group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA)),'SS']=group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA))]['SS']-4
            if int(ME)==2 and PA==4:
                revise(group1,-7,ME,PA)
                revise(group4,-7,ME,PA)
                # group1.loc[(group1['ME']==int(ME)) & (group1['PA']==int(PA)),'SS']=group1.loc[(group1['ME']==int(ME)) & (group1['PA']==int(PA))]['SS']-6
                # group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA)),'SS']=group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA))]['SS']-6
            if int(ME)==3 and PA==0:
                revise(group4,-10,ME,PA)
                # group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA)),'SS']=group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA))]['SS']-12
            if int(ME)==3 and PA==1:
                revise(group4,-5,ME,PA)
                # group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA)),'SS']=group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA))]['SS']-5
            if int(ME)==3 and PA==2:
                # revise(group1,-2,ME,PA)
                revise(group4,-5,ME,PA)
                # group1.loc[(group1['ME']==int(ME)) & (group1['PA']==int(PA)),'SS']=group1.loc[(group1['ME']==int(ME)) & (group1['PA']==int(PA))]['SS']-2
                # group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA)),'SS']=group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA))]['SS']-5
            if int(ME)==3 and PA==3:
                revise(group1,-5,ME,PA)
                # revise(group2,-2,ME,PA)
                revise(group4,-5,ME,PA)
                # group1.loc[(group1['ME']==int(ME)) & (group1['PA']==int(PA)),'SS']=group1.loc[(group1['ME']==int(ME)) & (group1['PA']==int(PA))]['SS']-7
                # group2.loc[(group2['ME']==int(ME)) & (group2['PA']==int(PA)),'SS']=group2.loc[(group2['ME']==int(ME)) & (group2['PA']==int(PA))]['SS']-2
                # group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA)),'SS']=group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA))]['SS']-8
            if int(ME)==3 and PA==4:
                revise(group4,-5,ME,PA)
                # group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA)),'SS']=group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA))]['SS']-6
            if int(ME)==4 and PA==0:
                None
                revise(group1,5,ME,PA)
                revise(group2,5,ME,PA)
                revise(group4,-10,ME,PA)
                # group1.loc[(group1['ME']==int(ME)) & (group1['PA']==int(PA)),'SS']=group1.loc[(group1['ME']==int(ME)) & (group1['PA']==int(PA))]['SS']+5
                # group2.loc[(group2['ME']==int(ME)) & (group2['PA']==int(PA)),'SS']=group2.loc[(group2['ME']==int(ME)) & (group2['PA']==int(PA))]['SS']+6
                # group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA)),'SS']=group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA))]['SS']-6
            if int(ME)==4 and PA==1:
                revise(group1,5,ME,PA)
                revise(group2,7,ME,PA)
                revise(group4,5,ME,PA)
                # group1.loc[(group1['ME']==int(ME)) & (group1['PA']==int(PA)),'SS']=group1.loc[(group1['ME']==int(ME)) & (group1['PA']==int(PA))]['SS']+2
                # group2.loc[(group2['ME']==int(ME)) & (group2['PA']==int(PA)),'SS']=group2.loc[(group2['ME']==int(ME)) & (group2['PA']==int(PA))]['SS']+6
                # group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA)),'SS']=group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA))]['SS']+3
            if int(ME)==4 and PA==2:
                revise(group1,7,ME,PA)
                revise(group2,5,ME,PA)
                revise(group4,5,ME,PA)
                # group1.loc[(group1['ME']==int(ME)) & (group1['PA']==int(PA)),'SS']=group1.loc[(group1['ME']==int(ME)) & (group1['PA']==int(PA))]['SS']+7
                # group2.loc[(group2['ME']==int(ME)) & (group2['PA']==int(PA)),'SS']=group2.loc[(group2['ME']==int(ME)) & (group2['PA']==int(PA))]['SS']+6
                # group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA)),'SS']=group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA))]['SS']+3
            if int(ME)==5 and PA==0:
                revise(group2,10,ME,PA)
                # group2.loc[(group2['ME']==int(ME)) & (group2['PA']==int(PA)),'SS']=group2.loc[(group2['ME']==int(ME)) & (group2['PA']==int(PA))]['SS']+10
            if int(ME)==6 and PA==0:
                revise(group1,-5,ME,PA)
                revise(group2,5,ME,PA)
                # revise(group4,-2,ME,PA)
                # group1.loc[(group1['ME']==int(ME)) & (group1['PA']==int(PA)),'SS']=group1.loc[(group1['ME']==int(ME)) & (group1['PA']==int(PA))]['SS']-9
                # group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA)),'SS']=group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA))]['SS']-3
            if int(ME)==6 and PA==1:
                # None
                revise(group1,10,ME,PA)
                revise(group2,10,ME,PA)
                # revise(group4,-3,ME,PA)
                # group1.loc[(group1['ME']==int(ME)) & (group1['PA']==int(PA)),'SS']=group1.loc[(group1['ME']==int(ME)) & (group1['PA']==int(PA))]['SS']+8
                # group2.loc[(group2['ME']==int(ME)) & (group2['PA']==int(PA)),'SS']=group2.loc[(group2['ME']==int(ME)) & (group2['PA']==int(PA))]['SS']+10
                # group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA)),'SS']=group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA))]['SS']-3
            if int(ME)==6 and PA==2:
                # None
                revise(group2,5,ME,PA)
                revise(group3,-5,ME,PA)
                # group2.loc[(group2['ME']==int(ME)) & (group2['PA']==int(PA)),'SS']=group2.loc[(group2['ME']==int(ME)) & (group2['PA']==int(PA))]['SS']+7
            if int(ME)==7 and PA==0:
                # revise(group1,-2,ME,PA)
                revise(group2,5,ME,PA)
                # group1.loc[(group1['ME']==int(ME)) & (group1['PA']==int(PA)),'SS']=group1.loc[(group1['ME']==int(ME)) & (group1['PA']==int(PA))]['SS']-2
                # group2.loc[(group2['ME']==int(ME)) & (group2['PA']==int(PA)),'SS']=group2.loc[(group2['ME']==int(ME)) & (group2['PA']==int(PA))]['SS']+4
            if int(ME)==7 and PA==1:
                revise(group1,-5,ME,PA)
                revise(group2,5,ME,PA)
                # group1.loc[(group1['ME']==int(ME)) & (group1['PA']==int(PA)),'SS']=group1.loc[(group1['ME']==int(ME)) & (group1['PA']==int(PA))]['SS']-7
                # group2.loc[(group2['ME']==int(ME)) & (group2['PA']==int(PA)),'SS']=group2.loc[(group2['ME']==int(ME)) & (group2['PA']==int(PA))]['SS']+4
            if int(ME)==8 and PA==0:
                # None
                revise(group1,15,ME,PA)
                revise(group2,20,ME,PA)
                # revise(group3,5,ME,PA)
                # revise(group4,-3,ME,PA)
                # group1.loc[(group1['ME']==int(ME)) & (group1['PA']==int(PA)),'SS']=group1.loc[(group1['ME']==int(ME)) & (group1['PA']==int(PA))]['SS']+7
                # group2.loc[(group2['ME']==int(ME)) & (group2['PA']==int(PA)),'SS']=group2.loc[(group2['ME']==int(ME)) & (group2['PA']==int(PA))]['SS']+18
                # group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA)),'SS']=group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA))]['SS']-3
            if int(ME)==8 and PA==1:
                revise(group2,5,ME,PA)
                # group2.loc[(group2['ME']==int(ME)) & (group2['PA']==int(PA)),'SS']=group2.loc[(group2['ME']==int(ME)) & (group2['PA']==int(PA))]['SS']+5
            if int(ME)==9 and PA==0:
                revise(group2,5,ME,PA)
                revise(group4,-3,ME,PA)
                # group2.loc[(group2['ME']==int(ME)) & (group2['PA']==int(PA)),'SS']=group2.loc[(group2['ME']==int(ME)) & (group2['PA']==int(PA))]['SS']+5
                # group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA)),'SS']=group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA))]['SS']-3
            if int(ME)==9 and PA==1:
                revise(group2,5,ME,PA)
            if int(ME)==9 and PA==2:
                revise(group4,-5,ME,PA)
                # group2.loc[(group2['ME']==int(ME)) & (group2['PA']==int(PA)),'SS']=group2.loc[(group2['ME']==int(ME)) & (group2['PA']==int(PA))]['SS']+5
            if int(ME)==9 and PA==3:
                revise(group2,10,ME,PA)
                # group2.loc[(group2['ME']==int(ME)) & (group2['PA']==int(PA)),'SS']=group2.loc[(group2['ME']==int(ME)) & (group2['PA']==int(PA))]['SS']+8
            if int(ME)==9 and PA==4:
                revise(group2,10,ME,PA)
                # group2.loc[(group2['ME']==int(ME)) & (group2['PA']==int(PA)),'SS']=group2.loc[(group2['ME']==int(ME)) & (group2['PA']==int(PA))]['SS']+8
            if int(ME)==9 and PA==5:
                revise(group4,-5,ME,PA)
                # group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA)),'SS']=group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA))]['SS']-3
            if int(ME)==10 and PA==0:
                revise(group1,-12,ME,PA)
                revise(group2,-5,ME,PA)
                # group1.loc[(group1['ME']==int(ME)) & (group1['PA']==int(PA)),'SS']=group1.loc[(group1['ME']==int(ME)) & (group1['PA']==int(PA))]['SS']-10
                # group2.loc[(group2['ME']==int(ME)) & (group2['PA']==int(PA)),'SS']=group2.loc[(group2['ME']==int(ME)) & (group2['PA']==int(PA))]['SS']-5
            if int(ME)==11 and PA==0:
                # revise(group2,15,ME,PA)
                group2.loc[(group2['ME']==int(ME)) & (group2['PA']==int(PA)),'SS']=group2.loc[(group2['ME']==int(ME)) & (group2['PA']==int(PA))]['SS']+5
            
            group1.loc[(group1['SS']<0),'SS']=0
            group2.loc[(group2['SS']>100),'SS']=100
            group2.loc[(group2['SS']<0),'SS']=0
            group3.loc[(group3['SS']<0),'SS']=0
            group4.loc[(group4['SS']<0),'SS']=0

            data={'g1':group1.loc[(group1['ME']==int(ME)) & (group1['PA']==int(PA))]['SS'],
                  'g2':group2.loc[(group2['ME']==int(ME)) & (group2['PA']==int(PA))]['SS'],
                  'g3':group3.loc[(group3['ME']==int(ME)) & (group3['PA']==int(PA))]['SS'],
                  'g4':group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA))]['SS']}
            data=pd.DataFrame(data)
            box=data.boxplot(showmeans=True,return_type='dict')
            plt.ylabel('SS')
            plt.savefig(f'./statistic/statistic_from_ave_{ME}_{PA}.jpg')
            plt.close()
            exception_cases+=exception_value(
                group1.loc[(group1['ME']==int(ME)) & (group1['PA']==int(PA))],
                group2.loc[(group2['ME']==int(ME)) & (group2['PA']==int(PA))],
                group3.loc[(group3['ME']==int(ME)) & (group3['PA']==int(PA))],
                group4.loc[(group4['ME']==int(ME)) & (group4['PA']==int(PA))],
                box)
            

            # g1_var=g1_ME['SS'].var()
            # g2_var=g2_ME['SS'].var()
            # g3_var=g3_ME['SS'].var()
            # g4_var=g4_ME['SS'].var()

            # g1_min=g1_ME['SS'].min()
            # g2_min=g2_ME['SS'].min()
            # g3_min=g3_ME['SS'].min()
            # g4_min=g4_ME['SS'].min()

            # g1_max=g1_ME['SS'].max()
            # g2_max=g2_ME['SS'].max()
            # g3_max=g3_ME['SS'].max()
            # g4_max=g4_ME['SS'].max()

            # g1_mode=g1_ME['SS'].quantile(0.5)
            # g2_mode=g2_ME['SS'].quantile(0.5)
            # g3_mode=g3_ME['SS'].quantile(0.5)
            # g4_mode=g4_ME['SS'].quantile(0.5)

            # g1_median=g1_ME['SS'].median()
            # g2_median=g2_ME['SS'].median()
            # g3_median=g3_ME['SS'].median()
            # g4_median=g4_ME['SS'].median()

            # df['MEAN']=[g1_mean,g2_mean,g3_mean,g4_mean]
            # df['VAR']=[g1_var,g2_var,g3_var,g4_var]
            # df['MIN']=[g1_min,g2_min,g3_min,g4_min]
            # df['MAX']=[g1_max,g2_max,g3_max,g4_max]
            # df['MODE']=[g1_mode,g2_mode,g3_mode,g4_mode]
            # df['MEDIAN']=[g1_median,g2_median,g3_median,g4_median]
            
            # df=pd.DataFrame(df)
            # df=df.round(2)
            # df.to_csv(f'./statistic/statistic_from_ave_{ME}_{PA}.csv',index=True)
    exception_cases=exception_cases[0].append(exception_cases[1:])
    exception_cases.sort_index(inplace=True)
    exception_cases.to_csv('./statistic/exception.csv',index=False)
    return pd.concat([group1,group2,group3,group4])

def statistic_revise(csv_dir='./result_ave.csv'):
    df=pd.read_csv(csv_dir)
    group1=df[0:2980]
    group2=df[2981:5781]
    group3=df[5782:8672]
    group4=df[8673:11542]
    df=do_group(group1,group2,group3,group4)
    df=df.round(2)
    df.to_csv('./result_ave_revise.csv',index=False)

def decreasing(scores):
    return all(x>=y for x, y in zip(scores, scores[1:]))

def revise_manual_preface(csv_dir='./result_ave_revise.csv'):
    df=pd.read_csv(csv_dir)
    for id in range(400):
        score=[]
        ME='09'
        for PA in PARAMETERs[ME]:
            try:
                score.append(df.loc[(df['ID']==id)&(df['ME']==int(ME))&(df['PA']==PA),'SS'].iloc[0])
            except:
                None
        # print(id,score)
        if len(score)>=2:
            if not decreasing(score):
                print(id)
                # # print(f'{id}')
                # print(False)

def scale(csv_dir='./result_ave_revise.csv'):
    None  

if __name__=='__main__':
    # convert_csv()
    # statistic_revise()
    revise_manual_preface()
                




