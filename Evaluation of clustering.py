import pandas as pd
import numpy as np
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn import metrics
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix

All_Tonimoto=pd.read_csv('H:/Uni Tese/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/New Result/Compare Cluster by tonimoto/Similarity(tonimoto)_Between_Chemical.csv',delimiter="\t")
All_Tonimoto.insert(5,'Dissimilarity','')
All_Tonimoto.Dissimilarity=1-All_Tonimoto.Tonimoto
All_Tonimoto.Dissimilarity=All_Tonimoto.Dissimilarity.astype("float32")

#DEG Only
Out=pd.read_csv('H:/Uni Tese/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/New Result/Q4/Clustering By Only Gene/All_Result.csv',delimiter="\t")
Out.loc[Out.PredictClass==12,'PredictClass']=1
Out.loc[Out.PredictClass==13,'PredictClass']=2
Out.loc[Out.PredictClass==14,'PredictClass']=3
Out.loc[Out.PredictClass==15,'PredictClass']=4

Out.PredictClass=Out.PredictClass.astype("int")
Out.rename(columns={'PredictClass': 'Cluster'}, inplace=True)
Out.insert(7,'ClassName','')
AllClass=pd.read_csv("H:/Uni Tese/Tese 2/All Drug_By Mesh _Class/All Drug_In Mesh_Thraputic_By_ClassCodeName.csv",delimiter="\t")
AllClass=AllClass[['ClassCode','ClassName']]
AllClass=AllClass.drop_duplicates().reset_index(drop=True)
for i in Out.index:
    if Out.at[i,'Class']!=-1:
        Out.at[i,'ClassName']=AllClass[AllClass.ClassCode==int(Out.at[i,'Class'])]['ClassName'].iloc[-1]
    else:
        Out.at[i,'ClassName']=-1


OutClass=Out[Out.Class!=-1]

#Chemical Only
Out=pd.read_csv('H:/Uni Tese/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/New Result/Q4/Clustering By Only Chemical/WithDuplicate/All_Result.csv',delimiter="\t")
Out.loc[Out.PredictClass==0,'PredictClass']=1
Out.loc[Out.PredictClass==1,'PredictClass']=2
Out.loc[Out.PredictClass==2,'PredictClass']=3
Out.loc[Out.PredictClass==3,'PredictClass']=4
Out.loc[Out.PredictClass==4,'PredictClass']=5
Out.loc[Out.PredictClass==5,'PredictClass']=6
Out.loc[Out.PredictClass==6,'PredictClass']=7
Out.loc[Out.PredictClass==8,'PredictClass']=8
Out.loc[Out.PredictClass==9,'PredictClass']=9
Out.loc[Out.PredictClass==10,'PredictClass']=10
Out.loc[Out.PredictClass==11,'PredictClass']=11
Out.loc[Out.PredictClass==12,'PredictClass']=12
Out.loc[Out.PredictClass==13,'PredictClass']=13
Out.loc[Out.PredictClass==14,'PredictClass']=14
Out.loc[Out.PredictClass==15,'PredictClass']=15


Out.PredictClass=Out.PredictClass.astype("int")
Out.rename(columns={'PredictClass': 'Cluster'}, inplace=True)
Out.insert(7,'ClassName','')
AllClass=pd.read_csv("H:/Uni Tese/Tese 2/All Drug_By Mesh _Class/All Drug_In Mesh_Thraputic_By_ClassCodeName.csv",delimiter="\t")
AllClass=AllClass[['ClassCode','ClassName']]
AllClass=AllClass.drop_duplicates().reset_index(drop=True)
for i in Out.index:
    if Out.at[i,'Class']!=-1:
        Out.at[i,'ClassName']=AllClass[AllClass.ClassCode==int(Out.at[i,'Class'])]['ClassName'].iloc[-1]
    else:
        Out.at[i,'ClassName']=-1

# Count[Count.index==1].iloc[0,0]
# Out.to_csv('H:/Uni Tese/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/Out_L4_ByClassNAme.csv',index=False,sep="\t")
    
OutClass=Out[Out.Class!=-1]


# Chemical+Gene    OurApproach
Out=pd.read_csv('H:/Uni Tese/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/Out_L4.csv',delimiter="\t")
Out.loc[Out.PredictClass==2,'PredictClass']=1
Out.loc[Out.PredictClass==3,'PredictClass']=2
Out.loc[Out.PredictClass==4,'PredictClass']=3
Out.loc[Out.PredictClass==5,'PredictClass']=4
Out.loc[Out.PredictClass==6,'PredictClass']=5
Out.loc[Out.PredictClass==7,'PredictClass']=6
Out.loc[Out.PredictClass==10,'PredictClass']=7
Out.loc[Out.PredictClass==11,'PredictClass']=8
Out.loc[Out.PredictClass==12,'PredictClass']=9
Out.loc[Out.PredictClass==13,'PredictClass']=10
Out.loc[Out.PredictClass==14,'PredictClass']=11
Out.loc[Out.PredictClass==15,'PredictClass']=12
Out.PredictClass=Out.PredictClass.astype("int")
Out.rename(columns={'PredictClass': 'Cluster'}, inplace=True)
Out.insert(7,'ClassName','')
AllClass=pd.read_csv("H:/Uni Tese/Tese 2/All Drug_By Mesh _Class/All Drug_In Mesh_Thraputic_By_ClassCodeName.csv",delimiter="\t")
AllClass=AllClass[['ClassCode','ClassName']]
AllClass=AllClass.drop_duplicates().reset_index(drop=True)
for i in Out.index:
    if Out.at[i,'Class']!='NoClass':
        Out.at[i,'ClassName']=AllClass[AllClass.ClassCode==int(Out.at[i,'Class'])]['ClassName'].iloc[-1]
    else:
        Out.at[i,'ClassName']='NoClass'

# Count[Count.index==1].iloc[0,0]
# Out.to_csv('H:/Uni Tese/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/Out_L4_ByClassNAme.csv',index=False,sep="\t")
    
OutClass=Out[Out.Class!='NoClass']
AllDrugClusterClass=pd.pivot_table(OutClass, values='CID', index=['Cluster'],columns=['ClassName'], aggfunc='count',fill_value=0)

AllDrugClusterClass.loc[1]=round(AllDrugClusterClass.loc[1]/Count[Count.index==1].iloc[0,0] * 100 ).astype("int")
AllDrugClusterClass.loc[2]=round(AllDrugClusterClass.loc[2]/Count[Count.index==2].iloc[0,0] * 100  ).astype("int")
AllDrugClusterClass.loc[3]=round(AllDrugClusterClass.loc[3]/Count[Count.index==3].iloc[0,0] * 100  ).astype("int")
AllDrugClusterClass.loc[4]=round(AllDrugClusterClass.loc[4]/Count[Count.index==4].iloc[0,0] * 100  ).astype("int")
AllDrugClusterClass.loc[5]=round(AllDrugClusterClass.loc[5]/Count[Count.index==5].iloc[0,0] * 100  ).astype("int")
AllDrugClusterClass.loc[6]=round(AllDrugClusterClass.loc[6]/Count[Count.index==6].iloc[0,0] * 100  ).astype("int")
AllDrugClusterClass.loc[7]=round(AllDrugClusterClass.loc[7]/Count[Count.index==7].iloc[0,0] * 100  ).astype("int")
AllDrugClusterClass.loc[8]=round(AllDrugClusterClass.loc[8]/Count[Count.index==8].iloc[0,0] * 100  ).astype("int")
AllDrugClusterClass.loc[9]=round(AllDrugClusterClass.loc[9]/Count[Count.index==9].iloc[0,0] * 100  ).astype("int")
AllDrugClusterClass.loc[10]=round(AllDrugClusterClass.loc[10]/Count[Count.index==10].iloc[0,0] * 100  ).astype("int")
AllDrugClusterClass.loc[11]=round(AllDrugClusterClass.loc[11]/Count[Count.index==11].iloc[0,0] * 100  ).astype("int")
AllDrugClusterClass.loc[12]=round(AllDrugClusterClass.loc[12]/Count[Count.index==12].iloc[0,0]   * 100  ).astype("int")

AllDrugClusterClass.to_csv('H:/Uni Tese/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/DrugCluster_By_DrugClassMESH_ByDetail.csv',sep="\t",index=True)

# OutClass.groupby('Cluster').count()
# OutClass = OutClass.drop(OutClass[OutClass.Cluster == 2].index)
# OutClass = OutClass.drop(OutClass[OutClass.Cluster == 3].index)
# OutClass = OutClass.drop(OutClass[OutClass.Cluster == 9].index)

# # normalized_mutual_info_score(OutClass.Class,OutClass.Cluster)
# adjusted_mutual_info_score(OutClass.Class,OutClass.Cluster)
# metrics.adjusted_rand_score(OutClass.Class,OutClass.Cluster)
# metrics.v_measure_score(OutClass.Class,OutClass.Cluster)
# metrics.fowlkes_mallows_score(OutClass.Class,OutClass.Cluster)
# contingency_matrix(OutClass.Class,OutClass.Cluster)

Result=np.ones((4,4),dtype='float32')
error=[]
for x , Cluster1 in zip(range(4) , Out.PredictClass.drop_duplicates()):
    print("x:" + str(x))
    for y , Cluster2 in zip(range(4) , Out.PredictClass.drop_duplicates()):
            
        print("y:" + str(y))
        Data1=Out[Out.PredictClass==Cluster1].reset_index(drop=True)
        Data2=Out[Out.PredictClass==Cluster2].reset_index(drop=True)
        Data1=Data1[Data1.CID.isin(All_Tonimoto.CID1)].dropna()
        Data2=Data2[Data2.CID.isin(All_Tonimoto.CID1)].dropna()
        AllData=Data1.append(Data2)[['CID']].drop_duplicates()
        Sum=0
        AllTonimotoSelected=All_Tonimoto[All_Tonimoto.CID1.isin(AllData.CID)].reset_index(drop=True)
        count=0
        for i in Data1.index:
            AllTonimotoData1=AllTonimotoSelected[AllTonimotoSelected.CID1==Data1.at[i,'CID']]
            if len(AllTonimotoData1)!=0:
                # if Cluster1==Cluster2:
                #     j=i+1
                for j in Data2.index:
                   print(str(i)+"-"+str(j))
                   try:
                       Sum=Sum+AllTonimotoData1[AllTonimotoData1.CID2==Data2.at[j,'CID']].Dissimilarity.iloc[-1]
                   except:
                       count=count+1
                       continue
        error.append(count)
        if len(Data1)==0 or len(Data2)==0:
                Result[x,y]=1   
        else:
                Result[x,y]=Sum/(len(Data1)*len(Data2))

pd.DataFrame(Result).to_csv('H:/Uni Tese/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/New Result/Compare Cluster by tonimoto/OnlyHormonizom.csv',sep="\t",index=False)
1-np.mean(np.array(Result))
    

resultOurApproach=pd.read_csv(('H:/Uni Tese/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/New Result/Compare Cluster by tonimoto/ChemicalHormonizomWithoutConcat.csv'),delimiter="\t")
resultMaccWithoutDuplicate=pd.read_csv('H:/Uni Tese/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/New Result/Compare Cluster by tonimoto/ChemicalWithoutDuplicate.csv',delimiter="\t")

resultMaccWithoutDuplicate=np.array(resultMaccWithoutDuplicate.iloc[:13,:13])
MaccCohission=np.mean(resultMaccWithoutDuplicate.diagonal())
MaccSeparation=(np.sum(resultMaccWithoutDuplicate)-np.sum(resultMaccWithoutDuplicate.diagonal()))/((13*13)-13)

resultOurApproach=np.array(resultOurApproach.iloc[:,:])
OurApproachCohission=np.mean(resultOurApproach.diagonal())
OurApproachSeparation=(np.sum(resultOurApproach)-np.sum(resultOurApproach.diagonal()))/((12*12)-12)

resultMaccHormonizomWithConcat=np.array(Result)
MaccHormonizomWithConcatCohission=np.mean(resultMaccHormonizomWithConcat.diagonal())
MaccHormonizomWithConcatSeparation=(np.sum(resultMaccHormonizomWithConcat)-np.sum(resultMaccHormonizomWithConcat.diagonal()))/((2*2)-2)

resultHormonizom=np.array(Result)
MaccHormonizomWithConcatCohission=np.mean(resultHormonizom.diagonal())
MaccHormonizomWithConcatSeparation=(np.sum(resultHormonizom)-np.sum(resultHormonizom.diagonal()))/((4*4)-4)


# AllDistance=pd.pivot_table(All_Tonimoto, values='Dissimilarity', index=['CID1'],columns=['CID2'], fill_value=0.5))

# metrics.silhouette_score(AllDistance,Out.predictedClass, metric='precompute')

# Out=pd.read_csv('H:/Uni Tese/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/New Result/Q4/Clustering By Only Chemical/WithoutDuplicate/All_Result.csv',delimiter="\t")
# metrics.silhouette_score(AllDistance,Out.predictedClass, metric='precompute')


# Out=pd.read_csv('H:/Uni Tese/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/New Result/Q4/Clustering By Only Chemical/WithDuplicate/All_Result.csv',delimiter="\t")
# metrics.silhouette_score(AllDistance,Out.predictedClass, metric='precompute')


# Out=pd.read_csv('H:/Uni Tese/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/New Result/Q4/Clustering By Only Concatenate Chemical_Gene/All_Result.csv',delimiter="\t")
# metrics.silhouette_score(AllDistance,Out.predictedClass, metric='precompute')


# Out=pd.read_csv('H:/Uni Tese/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/New Result/Q4/Clustering By Only Gene/All_Result.csv',delimiter="\t")
# metrics.silhouette_score(AllDistance,Out.predictedClass, metric='precompute')
AllDrugPath_OurApproach=pd.read_csv('H:/Uni Tese/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/AllDrugPathInEvenCluster.csv',delimiter="\t")
AllDrugPath_OurApproach.groupby(['PathwayCode'])

for Pathway in AllDrugPath_OurApproach.PathwayName:
    AllDrugInPathway['CID'].drop_duplicates()=AllDrugPath_OurApproach[(AllDrugPath_OurApproach.PathwayName=='Melanogenesis') & (AllDrugPath_OurApproach.Cluster==9)]
    AllDrugInPathway


AllDrugPath_OurApproach.groupby('Cluster').count()
AllDrugPath=pd.pivot_table(AllDrugPath_OurApproach, values='Gene', index=['Cluster'],columns=['PathwayName'], aggfunc='count',fill_value=0)
pd.DataFrame(AllDrugPath).to_csv('H:/Uni Tese/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/AllPathway_Name_ByCluster.csv',sep="\t",index=True)
ax = sns.heatmap(AllDrugPathDelLowPathwayVar, annot=False, fmt="g", linewidths=0.4, cmap="YlOrRd",vmin=0,vmax=2)
plt.show()

List=[]
for col in pd.DataFrame(AllDrugPath).columns:
#    sum=sum+np.var(inpX[col])
    List.append(np.var(pd.DataFrame(AllDrugPath)[col]))

List.sort()
MeanVar=np.mean(List)
VarVar=np.var(List)
q3=np.quantile(List,0.75)
thershold=q3

ListCol=[]
for col in pd.DataFrame(AllDrugPath).columns:
    if np.var(pd.DataFrame(AllDrugPath)[col]) < q3:
        ListCol.append(col)
        
AllDrugPathDelLowPathwayVar=pd.DataFrame(AllDrugPath).drop(ListCol, axis=1)  
# AllDrugPathDelLowPathwayVar.loc[0]=np.round_(AllDrugPathDelLowPathwayVar.loc[0]/8119 * 100 , decimals=2 )
# AllDrugPathDelLowPathwayVar.loc[1]=np.round(AllDrugPathDelLowPathwayVar.loc[1]/8306 * 100 , decimals=2 )
# AllDrugPathDelLowPathwayVar.loc[2]=np.round(AllDrugPathDelLowPathwayVar.loc[2]/6598 * 100 , decimals=2 )
# AllDrugPathDelLowPathwayVar.loc[3]=np.round(AllDrugPathDelLowPathwayVar.loc[3]/6432 * 100 , decimals=2 )
# AllDrugPathDelLowPathwayVar.loc[4]=np.round(AllDrugPathDelLowPathwayVar.loc[4]/7674 * 100 , decimals=2 )
# AllDrugPathDelLowPathwayVar.loc[5]=np.round(AllDrugPathDelLowPathwayVar.loc[5]/795 * 100 , decimals=2 )
# AllDrugPathDelLowPathwayVar.loc[6]=np.round(AllDrugPathDelLowPathwayVar.loc[6]/3021 * 100 , decimals=2 )
# AllDrugPathDelLowPathwayVar.loc[7]=np.round(AllDrugPathDelLowPathwayVar.loc[7]/3762 * 100 , decimals=2 )
# AllDrugPathDelLowPathwayVar.loc[8]=np.round(AllDrugPathDelLowPathwayVar.loc[8]/3167 * 100 , decimals=2 )
# AllDrugPathDelLowPathwayVar.loc[9]=np.round(AllDrugPathDelLowPathwayVar.loc[9]/16296 * 100 , decimals=2 )
# AllDrugPathDelLowPathwayVar.loc[10]=np.round(AllDrugPathDelLowPathwayVar.loc[10]/1275 * 100 , decimals=2 )
# AllDrugPathDelLowPathwayVar.loc[11]=np.round(AllDrugPathDelLowPathwayVar.loc[11]/5386   * 100 , decimals=2 )
AllDrugPathDelLowPathwayVar.to_csv('H:/Uni Tese/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/AllClusterPathway_Name_DelLowPathwayVar.csv',sep="\t",index=True)  
AllDrugPathDelLowPathwayVar=pd.read_csv('H:/Uni Tese/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/AllClusterPathway_Name_DelLowPathwayVar.csv',delimiter="\t")  

# AllDrugPath_Normalize=pd.DataFrame(AllDrugPath)
# Sum=np.sum(pd.DataFrame(AllDrugPath).iloc[0])
# AllDrugPath_Normalize.iloc[0:1,:]=AllDrugPath_Normalize.iloc[0:1,:]/Sum


########################  Target in even Cluster ##########################
AllDrugPathInEvenCluster=pd.read_csv('H:/Uni Tese/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/AllDrugPathInEvenCluster.csv',delimiter="\t")
AllDrugPathInEvenCluster[['Gene']]=np.abs(AllDrugPathInEvenCluster[['Gene']])
AllDrugPathInEvenCluster.to_csv('H:/Uni Tese/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/AllDrugPathInEvenCluster_WithoutUpOrDown.csv',index=False,sep="\t")

#### Omim Disease
Omim =pd.read_csv('C:/Users/aghil/Desktop/drug gene in clusters/Omim_Disease_0.txt',delimiter="\t")
AllOmim=pd.DataFrame(columns=Omim.columns)
AllOmim.insert(0,'Cluster',np.NaN)
for i in range(12):
    temp =pd.read_csv('C:/Users/aghil/Desktop/drug gene in clusters/Omim_Disease_'+ str(i) +'.txt',delimiter="\t")
    AllOmim=AllOmim.append(temp)
    AllOmim=AllOmim.fillna(i+1)
AllOmim.Cluster=AllOmim.Cluster.astype("int")
AllOmim.to_csv('C:/Users/aghil/Desktop/drug gene in clusters/All_Omim_Disease.txt',index=False,sep="\t")
   
#### Pathway
patway =pd.read_csv('C:/Users/aghil/Desktop/drug gene in clusters/patway_0.txt',delimiter="\t")
Allpatway=pd.DataFrame(columns=patway.columns)
Allpatway.insert(0,'Cluster',np.NaN)
for i in range(12):
    temp =pd.read_csv('C:/Users/aghil/Desktop/drug gene in clusters/patway_'+ str(i) +'.txt',delimiter="\t")
    Allpatway=Allpatway.append(temp)
    Allpatway=Allpatway.fillna(i+1)
Allpatway.Cluster=Allpatway.Cluster.astype("int")
Allpatway.to_csv('C:/Users/aghil/Desktop/drug gene in clusters/All_patway.txt',index=False,sep="\t")
     
#### GOTERM_BP
GOTERM_BP =pd.read_csv('C:/Users/aghil/Desktop/drug gene in clusters/GOTERM_BP_0.txt',delimiter="\t")
AllGOTERM_BP=pd.DataFrame(columns=GOTERM_BP.columns)
AllGOTERM_BP.insert(0,'Cluster',np.NaN)
for i in range(12):
    temp =pd.read_csv('C:/Users/aghil/Desktop/drug gene in clusters/GOTERM_BP_'+ str(i) +'.txt',delimiter="\t")
    AllGOTERM_BP=AllGOTERM_BP.append(temp)
    AllGOTERM_BP=AllGOTERM_BP.fillna(i+1)
AllGOTERM_BP.Cluster=AllGOTERM_BP.Cluster.astype("int")
AllGOTERM_BP.to_csv('C:/Users/aghil/Desktop/drug gene in clusters/AllGOTERM_BP.txt',index=False,sep="\t")
     
#### GOTERM_CC
GOTERM_CC =pd.read_csv('C:/Users/aghil/Desktop/drug gene in clusters/GOTERM_BP_0.txt',delimiter="\t")
AllGOTERM_CC=pd.DataFrame(columns=GOTERM_CC.columns)
AllGOTERM_CC.insert(0,'Cluster',np.NaN)
for i in range(12):
    temp =pd.read_csv('C:/Users/aghil/Desktop/drug gene in clusters/GOTERM_CC_'+ str(i) +'.txt',delimiter="\t")
    AllGOTERM_CC=AllGOTERM_CC.append(temp)
    AllGOTERM_CC=AllGOTERM_CC.fillna(i+1)
AllGOTERM_CC.Cluster=AllGOTERM_CC.Cluster.astype("int")
AllGOTERM_CC.to_csv('C:/Users/aghil/Desktop/drug gene in clusters/AllGOTERM_CC.txt',index=False,sep="\t")

#### GOTERM_MF

GOTERM_MF =pd.read_csv('C:/Users/aghil/Desktop/drug gene in clusters/GOTERM_MF_0.txt',delimiter="\t")
AllGOTERM_MF=pd.DataFrame(columns=GOTERM_MF.columns)
AllGOTERM_MF.insert(0,'Cluster',np.NaN)
for i in range(12):
    temp =pd.read_csv('C:/Users/aghil/Desktop/drug gene in clusters/GOTERM_MF_'+ str(i) +'.txt',delimiter="\t")
    AllGOTERM_MF=AllGOTERM_MF.append(temp)
    AllGOTERM_MF=AllGOTERM_MF.fillna(i+1)
AllGOTERM_MF.Cluster=AllGOTERM_MF.Cluster.astype("int")
AllGOTERM_MF.to_csv('C:/Users/aghil/Desktop/drug gene in clusters/AllGOTERM_MF.txt',index=False,sep="\t")
AllGOTERM_MF[AllGOTERM_MF.Cluster==1].groupby('Term').count()['Count']
