import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
# from matplotlib_venn import venn3, venn3_circles
from sklearn.metrics import jaccard_score
from matplotlib import rcParams
import re 
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import scipy.stats as stats
# import researchpy as rp
# import statsmodels.api as sm
# from statsmodels.formula.api import ols
# import pingouin as pg
    


ModelLayerForPredictClass="H:/Uni Tese/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final"
Data=pd.read_csv(str(ModelLayerForPredictClass)+"/Out_L4.csv",delimiter="\t")
Data.loc[Data.PredictClass== 2,'PredictClass'] = 1
Data.loc[Data.PredictClass== 3,'PredictClass'] = 2
Data.loc[Data.PredictClass== 4,'PredictClass'] = 3
Data.loc[Data.PredictClass== 5,'PredictClass'] = 4
Data.loc[Data.PredictClass== 6,'PredictClass'] = 5
Data.loc[Data.PredictClass== 7,'PredictClass'] = 6
Data.loc[Data.PredictClass== 10,'PredictClass'] = 7
Data.loc[Data.PredictClass== 11,'PredictClass'] = 8
Data.loc[Data.PredictClass== 12,'PredictClass'] = 9
Data.loc[Data.PredictClass== 13,'PredictClass'] = 10
Data.loc[Data.PredictClass== 14,'PredictClass'] = 11
Data.loc[Data.PredictClass== 15,'PredictClass'] = 12

Data.groupby(['PredictClass']).count()
Data.groupby(['Class']).count()

Hormonizom=pd.read_csv("/Users/aghil/Desktop/uni/Tese 2/final list/Train_By Common Multimodal Data/only hormonizom/each Cell Line Dose And Time In One Row/All_Hormonizom_Col_Lower_VarQuantile0.75_Remove.csv",delimiter="\t")

Path_Classifier_Result="/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/Macc(29074-156)_Hormonizom(29074-2086)_L12_L12-L110_L12_4_clustering_By_Only_RBM_By_Data_Remove_Macc_Col_Lower_Var0_Hormonizom_Lower_Quntile0.75/Classification Result"

#CID_0=pd.DataFrame(Data[Data['PredictClass']==0]['CID']).drop_duplicates()
#Hormonizom_Class_0=Hormonizom[Hormonizom.CID.isin(CID_0.CID)]

arrGene=Hormonizom.columns[4:]
################ Print Cluster 4 to 1*4
for j in range(2):
    figGroup,axesGroup=plt.subplots(2,3)
    plt.tight_layout()
    for i in range(3):
        k=i+(3*j)
        HormonizomClustre=Hormonizom[Hormonizom.CID.isin(pd.DataFrame(Data[Data['PredictClass']==k]['CID']).drop_duplicates().CID)]
        if len(HormonizomClustre)>0:
            gGroup=sns.heatmap(HormonizomClustre.iloc[:,4:].sort_values(by=list(arrGene)),ax=axesGroup[i])
            gGroup.set_title('Cluster '+str(k))
            gGroup.set_xlabel('DEG')
            gGroup.set_ylabel('Compund ID')
    figGroup = gGroup.get_figure()
    figGroup.savefig("/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/"+str(ModelLayerForPredictClass)+"/Hormonizom Cluster group "+str(j)+".png" )
    
################  print hormonizom cluster tak tak ##############
for i in range(16):
    HormonizomClustre=Hormonizom[Hormonizom.CID.isin(pd.DataFrame(Data[Data['PredictClass']==i]['CID']).drop_duplicates().CID)]
    if len(HormonizomClustre)>0:
        figTak=plt.figure()
        gTak=sns.heatmap(HormonizomClustre.iloc[:,4:].sort_values(by=list(arrGene)))
        gTak.set_title('Cluster '+str(i))
        gTak.set_xlabel('DEG')
        gTak.set_ylabel('Compund ID')
        #    figTak = gTak[i].get_figure()
        figTak.savefig(str(ModelLayerForPredictClass)+"/Hormonizom Cluster "+str(i)+".png" )

# np.count_nonzero(HormonizomClustre==1)
# np.count_nonzero(HormonizomClustre==0)

# i=2
# HormonizomClustre=Hormonizom[Hormonizom.CID.isin(pd.DataFrame(Data[Data['PredictClass']==i]['CID']).drop_duplicates().CID)]
# if len(HormonizomClustre)>0:
#    figTak=plt.figure()
#    gTak=sns.heatmap(HormonizomClustre.iloc[:,4:].sort_values(by=list(arrGene)))
#    gTak.set_title('Cluster '+str(i))
#    gTak.set_xlabel('DEG')
#    gTak.set_ylabel('Compund ID')
#    #    figTak = gTak[i].get_figure()
#    figTak.savefig(str(ModelLayerForPredictClass)+"/Hormonizom Cluster "+str(i)+".png" )


################  print hormonizom cluster sum of even cluster ##############

ClassGene=pd.DataFrame(columns=Hormonizom.columns[4:])
ClassGene.insert(0,'Class',np.empty)
allClass=Data['PredictClass'].drop_duplicates()
for Class in np.array(allClass,"int"):
    i=len(ClassGene)
    ClassGene.at[i,'Class']=Class
    AllPredictCID=pd.DataFrame(Data[Data['PredictClass']==Class]['CID']).drop_duplicates()
    AllDataClass=Hormonizom[Hormonizom.CID.isin(AllPredictCID.CID)].drop_duplicates()
    for gene in  Hormonizom.columns[4:]:
       ClassGene.at[i,gene]=np.count_nonzero(AllDataClass[gene]==1)-np.count_nonzero(AllDataClass[gene]==0)

ClassGene.fillna(value=np.nan, inplace=True)
sns.set(rc={'figure.figsize':(14,10)})
sns.set(font_scale=2) 
gSum=sns.heatmap(ClassGene.iloc[:,1:].sort_values(by=list(arrGene)),yticklabels=ClassGene.Class.sort_values())
gSum.set(xlabel='DEG Feature', ylabel='Cluster')
figSum = gSum.get_figure()
figSum.savefig(str(ModelLayerForPredictClass)+"/Heatmap Hormonizom sum of drug feature in even cluster_final.png" , dpi = 400)


########### For Chemical ##################
###########################################
Macc=pd.read_csv("/Users/aghil/Desktop/uni/Tese 2/final list/Train_By Common Multimodal Data/only hormonizom/each Cell Line Dose And Time In One Row/All_Macc_FP_Col_Var0_Remove.csv",delimiter="\t")
Macc.insert(0,'Class',Hormonizom['Class'])
Macc.rename(columns={'Name': 'CID'}, inplace=True)

arrChemical=Macc.columns[2:]
################ Print Cluster 4 to 1*4
for j in range(4):
    figGroup,axesGroup=plt.subplots(1,4)
    plt.tight_layout()
    for i in range(4):
        k=i+(4*j)
        MaccClustre=Macc[Macc.CID.isin(pd.DataFrame(Data[Data['PredictClass']==k]['CID']).drop_duplicates().CID)]
        if len(MaccClustre)>0:
            gGroup=sns.heatmap(MaccClustre.iloc[:,2:].sort_values(by=list(arrChemical)),ax=axesGroup[i])
            gGroup.set_title('Cluster '+str(k))
            gGroup.set_xlabel('DEG')
            gGroup.set_ylabel('Compund ID')
    figGroup = gGroup.get_figure()
    figGroup.savefig(str(ModelLayerForPredictClass)+"/Macc Cluster group "+str(j)+".png" )
    
################  print Chemical cluster tak tak ##############
# for i in range(16):
#     MaccClustre=Macc[Macc.CID.isin(pd.DataFrame(Data[Data['PredictClass']==i]['CID']).drop_duplicates().CID)]
#     if len(MaccClustre)>0:
        
#         figTak=plt.figure()  
#         gTak=sns.heatmap(MaccClustre.iloc[:,2:].sort_values(by=list(arrChemical)))
#         gTak.set_title('Cluster '+str(i))
#         gTak.set_xlabel('DEG')
#         gTak.set_ylabel('Compund ID')
#     #    figTak = gTak.get_figure()
#         figTak.savefig(str(ModelLayerForPredictClass)+"/Macc Cluster "+str(i)+".png" )



ClassFeature=pd.DataFrame(columns=Macc.columns[2:])
ClassFeature.insert(0,'Class',np.empty)
allClass=Data['PredictClass'].drop_duplicates()
for Class in np.array(allClass,"int"):
    i=len(ClassFeature)
    ClassFeature.at[i,'Class']=Class
    AllPredictCID=pd.DataFrame(Data[Data['PredictClass']==Class]['CID']).drop_duplicates()
    AllDataClass=Macc[Macc.CID.isin(AllPredictCID.CID)].drop_duplicates()
    for Feature in  Macc.columns[1:]:
       ClassFeature.at[i,Feature]=np.count_nonzero(AllDataClass[Feature]==1)-np.count_nonzero(AllDataClass[Feature]==0) 
  
ClassFeature.fillna(value=np.nan, inplace=True)
sns.set(rc={'figure.figsize':(14,10)})
sns.set(font_scale=2)  
gSum=sns.heatmap(ClassFeature.iloc[:,1:-1].sort_values(by=list(arrChemical)),yticklabels=ClassGene.Class.sort_values())
gSum.set(xlabel='Chemical Feature (MACC Keys)', ylabel='Cluster')
figSum = gSum.get_figure()
figSum.savefig(str(ModelLayerForPredictClass)+"/Heatmap Macc sum of drug feature in even cluster_final.png" , dpi=400 )

####### By MAccAnova Feature Only
macc=Macc[['Cluster','Name','MACCSFP28','MACCSFP31','MACCSFP43','MACCSFP53','MACCSFP57','MACCSFP83','MACCSFP137','MACCSFP149']]
ClassFeature=pd.DataFrame(columns=macc.columns[2:])
ClassFeature.insert(0,'Class',np.empty)
allClass=Data['PredictClass'].drop_duplicates()
for Class in np.array(allClass,"int"):
    i=len(ClassFeature)
    ClassFeature.at[i,'Class']=Class
    AllPredictCID=pd.DataFrame(Data[Data['PredictClass']==Class]['CID']).drop_duplicates()
    AllDataClass=macc[macc.Name.isin(AllPredictCID.CID)].drop_duplicates()
    for Feature in  macc.columns[2:]:
       ClassFeature.at[i,Feature]=np.count_nonzero(AllDataClass[Feature]==1)-np.count_nonzero(AllDataClass[Feature]==0) 
  
ClassFeature.fillna(value=np.nan, inplace=True)
sns.set(rc={'figure.figsize':(14,10)})
sns.set(font_scale=2)  
gSum=sns.heatmap(ClassFeature.iloc[:,1:].sort_values(by=list(macc.columns[2:])),yticklabels=ClassFeature.Class.sort_values())
gSum.set(xlabel='Chemical Feature (MACC Keys)', ylabel='Cluster')
figSum = gSum.get_figure()
# figSum.savefig(str(ModelLayerForPredictClass)+"/Heatmap Macc sum of drug feature in even cluster_final.png" , dpi=400 )

####### By HormonizomAnova Feature Only
hormonizom=Hormonizom[HormonizomAnova['HormonizomFeature']]
hormonizom.insert(0,'Cluster',L4.PredictClass.astype(int))
hormonizom.insert(1,'CID',Hormonizom.CID)
ClassGene=pd.DataFrame(columns=hormonizom.columns[2:])
ClassGene.insert(0,'Class',np.empty)
allClass=Data['PredictClass'].drop_duplicates()
for Class in np.array(allClass,"int"):
    i=len(ClassGene)
    ClassGene.at[i,'Class']=Class
    AllPredictCID=pd.DataFrame(Data[Data['PredictClass']==Class]['CID']).drop_duplicates()
    AllDataClass=hormonizom[hormonizom.CID.isin(AllPredictCID.CID)].drop_duplicates()
    for gene in  hormonizom.columns[2:]:
       ClassGene.at[i,gene]=np.count_nonzero(AllDataClass[gene]==1)-np.count_nonzero(AllDataClass[gene]==0)

ClassGene.fillna(value=np.nan, inplace=True)
sns.set(rc={'figure.figsize':(14,10)})
sns.set(font_scale=2) 
gSum=sns.heatmap(ClassGene.iloc[:,1:].sort_values(by=list(hormonizom.columns[2:])),yticklabels=ClassGene.Class.sort_values())
gSum.set(xlabel='DEG Feature', ylabel='Cluster')
figSum = gSum.get_figure()
# figSum.savefig(str(ModelLayerForPredictClass)+"/Heatmap Hormonizom sum of drug feature in even cluster_final.png" , dpi = 400)


#####  PCA macc Anova Selected significant Feature
del macc['Name']
from sklearn.decomposition import PCA
pca = PCA(4) 
pca.fit(macc.iloc[:,1:])   
pca_macc_data = pd.DataFrame(pca.transform(macc.iloc[:,1:])) 
pca_macc_data.columns=['Component1','Component2','Component3','Component4']
pca_macc_data.insert(0,'Cluster',macc['Cluster'])
pca_macc_data['Cluster'] = pca_macc_data['Cluster'].astype(int)
DF_pca_macc_data=pd.DataFrame(pca_macc_data.groupby('Cluster').mean())
DF_pca_macc_data.insert(0,'Cluster',DF_pca_macc_data.index)
DF_pca_macc_data=DF_pca_macc_data.reset_index(drop=True)
DFFinal_pca_macc_data=pd.melt(DF_pca_macc_data, id_vars="Cluster", var_name="Component", value_name="Value")
sns.set(rc={'figure.figsize':(28,40)})
sns.set(font_scale=1) 
ax=sns.factorplot(x='Cluster', y='Value', hue='Component', data=DFFinal_pca_macc_data, kind='bar',legend_out=False)
ax.set(xlabel='Cluster', ylabel='')
 
#####  PCA Hormonizom Anova Selected significant Feature
del hormonizom['CID']
from sklearn.decomposition import PCA
pca = PCA(4) 
pca.fit(hormonizom.iloc[:,1:])   
pca_hormonizom_data = pd.DataFrame(pca.transform(hormonizom.iloc[:,1:])) 
pca_hormonizom_data.columns=['Component1','Component2','Component3','Component4']
pca_hormonizom_data.insert(0,'Cluster',hormonizom['Cluster'])
pca_hormonizom_data['Cluster'] = pca_hormonizom_data['Cluster'].astype(int)
DF_pca_hormonizom_data=pd.DataFrame(pca_hormonizom_data.groupby('Cluster').mean())
DF_pca_hormonizom_data.insert(0,'Cluster',DF_pca_hormonizom_data.index)
DF_pca_hormonizom_data=DF_pca_hormonizom_data.reset_index(drop=True)
DFFinal_pca_hormonizom_data=pd.melt(DF_pca_hormonizom_data, id_vars="Cluster", var_name="Component", value_name="Value")
sns.set(rc={'figure.figsize':(28,40)})
sns.set(font_scale=1) 
ax=sns.factorplot(x='Cluster', y='Value', hue='Component', data=DFFinal_pca_hormonizom_data, kind='bar',legend_out=False)
ax.set(xlabel='Cluster', ylabel='')
 
L4[L4.PredictClass==12]['CID'].drop_duplicates()

############### Concat Chemical and Hormonizome 7 cluster ##############
##############################################################
AllArr=Macc.columns[2:]
AllArr=AllArr.append(Hormonizom.columns[4:])

Data.loc[Data.PredictClass== 1,'PredictClass'] = 'a'
Data.loc[Data.PredictClass== 2,'PredictClass'] = 'b'
Data.loc[Data.PredictClass== 4,'PredictClass'] = 'c'
Data.loc[Data.PredictClass== 6,'PredictClass'] = 'a'
Data.loc[Data.PredictClass== 11,'PredictClass'] = 'd'
Data.loc[Data.PredictClass== 12,'PredictClass'] = 'd'
Data.loc[Data.PredictClass== 8,'PredictClass'] = 'e'
Data.loc[Data.PredictClass== 7,'PredictClass'] = 'e'
Data.loc[Data.PredictClass== 10,'PredictClass'] = 'f'
Data.loc[Data.PredictClass== 5,'PredictClass'] = 'f'
Data.loc[Data.PredictClass== 9,'PredictClass'] = 'g'
Data.loc[Data.PredictClass== 3,'PredictClass'] = 'g'

AllData=pd.concat([Hormonizom.iloc[:,4:],Macc.iloc[:,2:]],axis=1)
AllData.insert(0,'CID',Hormonizom.CID)
ClassFeature=pd.DataFrame(columns=AllArr)
ClassFeature.insert(0,'Class',np.empty)
allClass=Data['PredictClass'].drop_duplicates()
for Class in np.array(allClass):
    i=len(ClassFeature)
    ClassFeature.at[i,'Class']=Class
    AllPredictCID=pd.DataFrame(Data[Data['PredictClass']==Class]['CID']).drop_duplicates()
    AllDataClass=AllData[AllData.CID.isin(AllPredictCID.CID)].drop_duplicates()
    for Feature in  AllData.columns[1:]:
       ClassFeature.at[i,Feature]=np.count_nonzero(AllDataClass[Feature]==1)-np.count_nonzero(AllDataClass[Feature]==0)

ClassFeature['Class']=ClassFeature['Class'].astype("int")
ClassFeature.fillna(value=np.nan, inplace=True)
sns.set(rc={'figure.figsize':(14,10)})
sns.set(font_scale=2)  
gSum=sns.heatmap(ClassFeature.iloc[:,:].sort_values(by=list(ClassFeature.index),axis=1),yticklabels=ClassFeature.Class.sort_values())
gSum.set(xlabel='Chemical and DEG feature', ylabel='Cluster')
fig = gSum.get_figure()
gSum.set_ylim(12, 0.1)

fig.savefig(str(ModelLayerForPredictClass)+"/Heatmap Macc And Hormonizom sum of drug feature in even cluster_final_After Merge_7Class.eps",format='eps' )

##############  Concat Macc and Hormonizom 5 Cluster #############
AllArr=Macc.columns[2:]
AllArr=AllArr.append(Hormonizom.columns[4:])

Data.loc[Data.PredictClass== 1,'PredictClass'] = 'A'
Data.loc[Data.PredictClass== 2,'PredictClass'] = 'B'
Data.loc[Data.PredictClass== 4,'PredictClass'] = 'C'
Data.loc[Data.PredictClass== 6,'PredictClass'] = 'A'
Data.loc[Data.PredictClass== 11,'PredictClass'] = 'D'
Data.loc[Data.PredictClass== 12,'PredictClass'] = 'D'
Data.loc[Data.PredictClass== 8,'PredictClass'] = 'C'
Data.loc[Data.PredictClass== 7,'PredictClass'] = 'C'
Data.loc[Data.PredictClass== 10,'PredictClass'] = 'E'
Data.loc[Data.PredictClass== 5,'PredictClass'] = 'E'
Data.loc[Data.PredictClass== 9,'PredictClass'] = 'E'
Data.loc[Data.PredictClass== 3,'PredictClass'] = 'E'


AllData=pd.concat([Hormonizom.iloc[:,4:],Macc.iloc[:,2:]],axis=1)
AllData.insert(0,'CID',Hormonizom.CID)
ClassFeature=pd.DataFrame(columns=AllArr)
ClassFeature.insert(0,'Class',np.empty)
allClass=Data['PredictClass'].drop_duplicates()
for Class in np.array(allClass):
    i=len(ClassFeature)
    ClassFeature.at[i,'Class']=Class
    AllPredictCID=pd.DataFrame(Data[Data['PredictClass']==Class]['CID']).drop_duplicates()
    AllDataClass=AllData[AllData.CID.isin(AllPredictCID.CID)].drop_duplicates()
    for Feature in  AllData.columns[1:]:
       ClassFeature.at[i,Feature]=np.count_nonzero(AllDataClass[Feature]==1)-np.count_nonzero(AllDataClass[Feature]==0)


ClassFeature.fillna(value=np.nan, inplace=True)
sns.set(rc={'figure.figsize':(14,10)})
sns.set(font_scale=2) 
g=sns.heatmap(ClassFeature.iloc[:,1:].sort_values(by=list(AllArr)),yticklabels=ClassFeature.Class.sort_values())
g.set(xlabel='Chemical and DEG feature', ylabel='Super Cluster')
fig = g.get_figure()
fig.savefig(str(ModelLayerForPredictClass)+"/Heatmap Macc And Hormonizom sum of drug feature in even cluster_final_After Merge_5Class.eps",format='eps' )

##################################################################
#################### All Decission Data ##########################

Conf_Matrix_Descission_1=pd.read_csv("/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/Macc(29074-156)_Hormonizom(29074-2086)_L12_L12-L110_L12_4_clustering_By_Only_RBM_By_Data_Remove_Macc_Col_Lower_Var0_Hormonizom_Lower_Quntile0.75/Classification Result/Confution_Matrixc_Decission1.txt",delimiter="\t")
Conf_Matrix_Descission_2=pd.read_csv("/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/Macc(29074-156)_Hormonizom(29074-2086)_L12_L12-L110_L12_4_clustering_By_Only_RBM_By_Data_Remove_Macc_Col_Lower_Var0_Hormonizom_Lower_Quntile0.75/Classification Result/Confution_Matrixc_Decission2.txt",delimiter="\t")
Conf_Matrix_Descission_3=pd.read_csv("/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/Macc(29074-156)_Hormonizom(29074-2086)_L12_L12-L110_L12_4_clustering_By_Only_RBM_By_Data_Remove_Macc_Col_Lower_Var0_Hormonizom_Lower_Quntile0.75/Classification Result/Confution_Matrixc_Decission3.txt",delimiter="\t")
Conf_Matrix_Descission_4=pd.read_csv("/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/Macc(29074-156)_Hormonizom(29074-2086)_L12_L12-L110_L12_4_clustering_By_Only_RBM_By_Data_Remove_Macc_Col_Lower_Var0_Hormonizom_Lower_Quntile0.75/Classification Result/Confution_Matrixc_Decission4.txt",delimiter="\t")
Conf_Matrix_Descission_5=pd.read_csv("/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/Macc(29074-156)_Hormonizom(29074-2086)_L12_L12-L110_L12_4_clustering_By_Only_RBM_By_Data_Remove_Macc_Col_Lower_Var0_Hormonizom_Lower_Quntile0.75/Classification Result/Confution_Matrixc_Decission5.txt",delimiter="\t")

Conf_Matrix_Descission_1=Conf_Matrix_Descission_1.drop_duplicates()
Conf_Matrix_Descission_2=Conf_Matrix_Descission_2.drop_duplicates()
Conf_Matrix_Descission_3=Conf_Matrix_Descission_3.drop_duplicates()
Conf_Matrix_Descission_4=Conf_Matrix_Descission_4.drop_duplicates()
Conf_Matrix_Descission_5=Conf_Matrix_Descission_5.drop_duplicates()

All_Confusion=Conf_Matrix_Descission_1.append(Conf_Matrix_Descission_2, ignore_index=True)
All_Confusion=All_Confusion.append(Conf_Matrix_Descission_3, ignore_index=True)
All_Confusion=All_Confusion.append(Conf_Matrix_Descission_4, ignore_index=True)
All_Confusion=All_Confusion.append(Conf_Matrix_Descission_5, ignore_index=True)

AllClass=pd.read_csv("/Users/aghil/Desktop/uni/Tese 2/All Drug_By Mesh _Class/All Drug_In Mesh_Thraputic_By_ClassCodeName.csv",delimiter="\t")
AllClass=AllClass[['ClassCode','ClassName']]
AllClass=AllClass.drop_duplicates().reset_index(drop=True)

All_Confusion=All_Confusion.drop_duplicates()
All_Confusion['RealClass'] = All_Confusion['RealClass'].map({1:AllClass.loc[0,'ClassName'],2:AllClass.loc[1,'ClassName'],3:AllClass.loc[2,'ClassName']
            ,4:AllClass.loc[3,'ClassName'],5:AllClass.loc[4,'ClassName'],6:AllClass.loc[5,'ClassName']
            ,7:AllClass.loc[6,'ClassName'],8:AllClass.loc[7,'ClassName'],9:AllClass.loc[8,'ClassName']
            ,10:AllClass.loc[9,'ClassName'],11:AllClass.loc[10,'ClassName'],12:AllClass.loc[11,'ClassName']
            ,13:AllClass.loc[12,'ClassName'],14:AllClass.loc[13,'ClassName'],15:AllClass.loc[14,'ClassName']
            ,16:AllClass.loc[15,'ClassName'],17:AllClass.loc[16,'ClassName']})

All_Confusion['PredictClass'] = All_Confusion['PredictClass'].map({1:AllClass.loc[0,'ClassName'],2:AllClass.loc[1,'ClassName'],3:AllClass.loc[2,'ClassName']
            ,4:AllClass.loc[3,'ClassName'],5:AllClass.loc[4,'ClassName'],6:AllClass.loc[5,'ClassName']
            ,7:AllClass.loc[6,'ClassName'],8:AllClass.loc[7,'ClassName'],9:AllClass.loc[8,'ClassName']
            ,10:AllClass.loc[9,'ClassName'],11:AllClass.loc[10,'ClassName'],12:AllClass.loc[11,'ClassName']
            ,13:AllClass.loc[12,'ClassName'],14:AllClass.loc[13,'ClassName'],15:AllClass.loc[14,'ClassName']
            ,16:AllClass.loc[15,'ClassName'],17:AllClass.loc[16,'ClassName']})
    
np.set_printoptions(suppress=True)    
pivot=All_Confusion.pivot_table(index="RealClass",columns='PredictClass',values='CID',aggfunc='count')
pivot=pivot.fillna(0)

plt.figure(figsize=(14,12),dpi=80)
ax=sns.heatmap(pivot,cmap="coolwarm",annot=True,fmt='g')
ax.set_ylim(14, 0.1)

All_Confusion.CID.drop_duplicates().to_csv("/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/Macc(29074-156)_Hormonizom(29074-2086)_L12_L12-L110_L12_4_clustering_By_Only_RBM_By_Data_Remove_Macc_Col_Lower_Var0_Hormonizom_Lower_Quntile0.75/Classification Result/All_Predict_CID.txt",index=False)

All_Prediction_CID_NameFromPubchem=pd.read_csv("/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/Macc(29074-156)_Hormonizom(29074-2086)_L12_L12-L110_L12_4_clustering_By_Only_RBM_By_Data_Remove_Macc_Col_Lower_Var0_Hormonizom_Lower_Quntile0.75/Classification Result/PubChem_compound_list_Predict.csv",delimiter=",")
#All_Wrong_Prediction_CID_NameFromPubchem.at[1,'cmpdname']

All_Confusion.insert(0,'cmpdname',' ')
All_Confusion=All_Confusion.reset_index(drop=True)
for i in range(All_Confusion.shape[0]):
    compundName=All_Prediction_CID_NameFromPubchem[All_Prediction_CID_NameFromPubchem.cid==All_Confusion.at[i,'CID']].index
    All_Confusion.at[i,'cmpdname']=All_Prediction_CID_NameFromPubchem.at[compundName[0],'cmpdname']
All_Confusion.to_csv(str(Path_Classifier_Result)+"/All_Confusion_Matrix.csv",sep="\t",index=False)


All_Wrong_Prediction=All_Confusion[All_Confusion['PredictClass']!=All_Confusion['RealClass']]
All_Wrong_Prediction.to_csv(str(Path_Classifier_Result)+"/All_Wrong_Prediction.csv",sep="\t",index=False)

All_Correct_Prediction=All_Confusion[All_Confusion['PredictClass']==All_Confusion['RealClass']]
All_Correct_Prediction.to_csv(str(Path_Classifier_Result)+"/All_Correct_Prediction.csv",sep="\t",index=False)

CorrectAndWrong=All_Wrong_Prediction[All_Wrong_Prediction.CID.isin(All_Correct_Prediction.CID)] #drug in correct prediction simultunosly in wrong prediction
CorrectAndWrong.to_csv(str(Path_Classifier_Result)+"/CorrectAndWrong_Prediction.csv",sep="\t",index=False)


#Conf_Matrix_Descission_4[Conf_Matrix_Descission_4.CID.isin(Conf_Matrix_Descission_5.CID)]


CID_HC=pd.DataFrame(All_Wrong_Prediction.CID.drop_duplicates())
All_Wrong_Prediction_HCS=pd.read_csv("/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical_SE/Macc(2134-149)_Hormonizom(2134-2072)_SE(2134-1465)_L12_L12-L110_L12_4_Classification_Macc_Col_Lower_Var0_Hormonizom_And_SE_Lower_Quntile0.75/Classification Result/All_Wrong_Prediction.csv",delimiter="\t")

common=All_Wrong_Prediction_HCS[~All_Wrong_Prediction_HCS.CID.isin(All_Wrong_Prediction.CID)].reset_index(drop=True)
common.to_csv("/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical_SE/Macc(2134-149)_Hormonizom(2134-2072)_SE(2134-1465)_L12_L12-L110_L12_4_Classification_Macc_Col_Lower_Var0_Hormonizom_And_SE_Lower_Quntile0.75/Classification Result/All_Wrong_Prediction_Only_In_HCS.csv",sep="\t",index=False)

#################### Antiviral drug ###################
L4=pd.read_csv('/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/Out_L4.csv',delimiter="\t")
L4.loc[L4.PredictClass== 2,'PredictClass'] = 1
L4.loc[L4.PredictClass== 3,'PredictClass'] = 2
L4.loc[L4.PredictClass== 4,'PredictClass'] = 3
L4.loc[L4.PredictClass== 5,'PredictClass'] = 4
L4.loc[L4.PredictClass== 6,'PredictClass'] = 5
L4.loc[L4.PredictClass== 7,'PredictClass'] = 6
L4.loc[L4.PredictClass== 10,'PredictClass'] = 7
L4.loc[L4.PredictClass== 11,'PredictClass'] = 8
L4.loc[L4.PredictClass== 12,'PredictClass'] = 9
L4.loc[L4.PredictClass== 13,'PredictClass'] = 10
L4.loc[L4.PredictClass== 14,'PredictClass'] = 11
L4.loc[L4.PredictClass== 15,'PredictClass'] = 12

L4_WithoutNoClass=L4[L4.Class!='NoClass'].reset_index(drop=True)
L4.PredictClass.drop_duplicates()
L4=L4.iloc[:,0:3]
# L4=L4.drop_duplicates()
Hydroxychloroquine=L4[L4.CID==3652]
Nitazoxanide=L4[L4.CID==41684]  
disulfiram=L4[L4.CID==3117]
lopinavir=L4[L4.CID==92727]

Remdesivir=L4[L4.CID==121304016]  
galidesivir=L4[L4.CID==10445549 ]  
Favipiravir=L4[L4.CID==492405]  
Baloxavir=L4[L4.CID==124081876]  
Marboxil=L4[L4.CID==124081896]
Ribavirin=L4[L4.CID==37542]  
TENOFOVIR_ALAFENAMIDE=L4[L4.CID==9574768] 
Galidesivir=L4[L4.CID==10445549] 
ritonavir=L4[L4.CID==392622]
Interferon_alfa_2a=L4[L4.CID==46506712]



tonimoto=pd.read_csv('/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Chemical/Similarity(tonimoto)_Between_Chemical.csv',delimiter="\t")

ax=sns.distplot(tonimoto.Tonimoto)
ax.set_xlim(0,1)
ax.set(xlabel='Drug pair similarity by chemical structure feature')
plt.show()

plt.figure(figsize=(2,5),dpi=80)
ax=sns.boxplot(y=tonimoto.Tonimoto)
# ax.set_xlim(0,1)
ax.set(xlabel='Drug pair similarity by chemical structure feature')
plt.show()


# BestSimilarTo_Hydroxychloroquine=tanimoto[(tanimoto.CID1==3652) & (tanimoto.Tonimoto > 0.60)]['CID2']
# for i in BestSimilarTo_Hydroxychloroquine.index:
#     print(str(i))
#     print(str(L4[L4.CID==BestSimilarTo_Hydroxychloroquine[i]]))

L4_Clustre5=L4[L4.PredictClass==5]
L4_Clustre3=L4[L4.PredictClass==3]
L4_Clustre4=L4[L4.PredictClass==4]
L4_Clustre10=L4[L4.PredictClass==10]

L4_Clustre5=L4_Clustre5.drop_duplicates()
L4_Clustre3=L4_Clustre3.drop_duplicates()
L4_Clustre4=L4_Clustre4.drop_duplicates()
L4_Clustre10=L4_Clustre10.drop_duplicates()

L4_Clustre5.to_csv('/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/L4_Clustre5(Hydroxychloroquine).csv',sep="\t",index=False)
L4_Clustre3.to_csv('/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/L4_Clustre3(disulfiram_lopinavir).csv',sep="\t",index=False)
L4_Clustre4.to_csv('/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/L4_Clustre4(Nitazoxanide).csv',sep="\t",index=False)
L4_Clustre10.to_csv('/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/L4_Clustre10(Nitazoxanide_disulfiram_lopinavir).csv',sep="\t",index=False)


# L4_Clustre5_Hydroxychloroquine = set(L4_Clustre5['CID'])
# L4_Clustre3_disulfiram_lopinavir = set(L4_Clustre3['CID'])
# L4_Clustre4_Nitazoxanide = set(L4_Clustre4['CID'])
# L4_Clustre10_Nitazoxanide_disulfiram_lopinavir = set(L4_Clustre10['CID'])
# venn3([L4_Clustre5_Hydroxychloroquine, L4_Clustre3_disulfiram_lopinavir, L4_Clustre4_Nitazoxanide], ('L4_Clustre5_Hydroxychloroquine', 'L4_Clustre3_disulfiram_lopinavir', 'L4_Clustre4_Nitazoxanide'))
# plt.show()

List1=L4_Clustre5[L4_Clustre5.CID.isin(L4_Clustre3.CID)].dropna()
List2=L4_Clustre10[L4_Clustre10.CID.isin(L4_Clustre4.CID)].dropna()
CommonDrug=List1[List1.CID.isin(List2.CID)].dropna()
CommonDrug.to_csv("/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/L4_CommonDrug_In_4Cluster.csv",sep="\t",index=False)
CommonDrug['CID'].to_csv("/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/L4_CommonDrug_Pubchem.csv",sep="\t",index=False)

Pubchem_Common_Drug=pd.read_csv("/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/L4_CommonDrug_In_Pubchem.csv",delimiter=",")
Pubchem_Common_Drug.to_csv("/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/L4_CommonDrug_In_Pubchem.csv",sep="\t",index=False)
Pubchem_Common_Drug_cmpName=Pubchem_Common_Drug[['cid','cmpdname','inchikey']]


tonimoto=pd.read_csv('/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Chemical/Similarity(tonimoto)_Between_Chemical.csv',delimiter="\t")
Cluster5_tonimoto=tonimoto[tonimoto.CID2.isin(L4_Clustre5.CID)]
Cluster3_10_tonimoto=tonimoto[tonimoto.CID2.isin(L4_Clustre3.CID.append(L4_Clustre10.CID))]
Cluster4_10_tonimoto=tonimoto[tonimoto.CID2.isin(L4_Clustre4.CID.append(L4_Clustre10.CID))]
# Cluster13_tonimoto=tonimoto[tonimoto.CID2.isin(L4_Clustre13.CID)]

TonimotoSimilarity_Hydroxychloroquine=Cluster5_tonimoto[(Cluster5_tonimoto.CID1==3652) & (Cluster5_tonimoto.Tonimoto>0.6)].sort_values(by='Tonimoto', ascending=False)
TonimotoSimilarity_disulfiram=Cluster3_10_tonimoto[(Cluster3_10_tonimoto.CID1==3117) & (Cluster3_10_tonimoto.Tonimoto>0.6)].sort_values(by='Tonimoto', ascending=False)
TonimotoSimilarity_lopinavir=Cluster3_10_tonimoto[(Cluster3_10_tonimoto.CID1==92727) & (Cluster3_10_tonimoto.Tonimoto>0.7)].sort_values(by='Tonimoto', ascending=False)
TonimotoSimilarity_Nitazoxanide=Cluster4_10_tonimoto[(Cluster4_10_tonimoto.CID1==41684) & (Cluster4_10_tonimoto.Tonimoto>0.6)].sort_values(by='Tonimoto', ascending=False)



# TonimotoSimilarity_Hydroxychloroquine[['CID1','CID2']]
# TonimotoSimilarity_lopinavir[['CID1','CID2']]
# TonimotoSimilarity_Nitazoxanide[['CID1','CID2']]
# TonimotoSimilarity_disulfiram[['CID1','CID2']]

# _Hydroxychloroquine = set(Cluster6_tonimoto_Hydroxychloroquine['CID2'])
# _disulfiram_lopinavir = set(Cluster4_tonimoto_disulfiram_lopinavir['CID2'])
# _Nitazoxanide = set(Cluster5_tonimoto_Nitazoxanide['CID2'])
# _Nitazoxanide_disulfiram_lopinavir = set(Cluster13_tonimoto_Nitazoxanide_disulfiram_lopinavir['CID2'])
# venn3([_Hydroxychloroquine, _disulfiram_lopinavir, _Nitazoxanide], ('L4_Clustre6_Hydroxychloroquine', 'L4_Clustre4_disulfiram_lopinavir', 'L4_Clustre5_Nitazoxanide'))
# plt.show()
############## Hormonizom similarity #########################

Similarity_Between_Hormonizom_CovidDrugCandid=pd.read_csv('/Users/aghil/Desktop/uni/Tese 2/Implementation/Similarity(OurSimilarityMeasure)_Between_Hormonizom_CovidDrugCandid.csv',delimiter="\t")
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
StandardHormonizomSimilarity = min_max_scaler.fit_transform(pd.DataFrame(Similarity_Between_Hormonizom_CovidDrugCandid['Similarity']))
Similarity_Between_Hormonizom_CovidDrugCandid.insert(7,'SimilarityNormal',np.round(StandardHormonizomSimilarity,2))

ax=sns.distplot(StandardHormonizomSimilarity)
ax.set(xlabel='Drug pairs Similarity By DEGs feature',ylabel='Our similarity measure')
plt.show()

plt.figure(figsize=(2,5),dpi=80)
ax=sns.boxplot(y=StandardHormonizomSimilarity)
# ax.set_xlim(0,1)
ax.set(xlabel='Drug pairs Similarity By DEGs feature',ylabel='Our similarity measure')
plt.show()

Cluster5_HormonizomSimilarity=Similarity_Between_Hormonizom_CovidDrugCandid[Similarity_Between_Hormonizom_CovidDrugCandid.CID2.isin(L4_Clustre5.CID)]
Cluster3_10_HormonizomSimilarity=Similarity_Between_Hormonizom_CovidDrugCandid[Similarity_Between_Hormonizom_CovidDrugCandid.CID2.isin(L4_Clustre3.CID.append(L4_Clustre10.CID))]
Cluster4_10_HormonizomSimilarity=Similarity_Between_Hormonizom_CovidDrugCandid[Similarity_Between_Hormonizom_CovidDrugCandid.CID2.isin(L4_Clustre4.CID.append(L4_Clustre10.CID))]
# Cluster13_HormonizomSimilarity=Similarity_Between_Hormonizom_CovidDrugCandid[Similarity_Between_Hormonizom_CovidDrugCandid.CID2.isin(L4_Clustre13.CID)]


HormonizomSimilarity_Hydroxychloroquine=Cluster5_HormonizomSimilarity[(Cluster5_HormonizomSimilarity.CID1==3652) & (Cluster5_HormonizomSimilarity.SimilarityNormal>0.63) ].sort_values(by='SimilarityNormal', ascending=False)
HormonizomSimilarity_disulfiram=Cluster3_10_HormonizomSimilarity[(Cluster3_10_HormonizomSimilarity.CID1==3117) & (Cluster3_10_HormonizomSimilarity.SimilarityNormal>0.64)].sort_values(by='SimilarityNormal', ascending=False)
HormonizomSimilarity_lopinavir=Cluster3_10_HormonizomSimilarity[(Cluster3_10_HormonizomSimilarity.CID1==92727)& (Cluster3_10_HormonizomSimilarity.SimilarityNormal>0.64)].sort_values(by='SimilarityNormal', ascending=False)
HormonizomSimilarity_Nitazoxanide=Cluster4_10_HormonizomSimilarity[(Cluster4_10_HormonizomSimilarity.CID1==41684) & (Cluster4_10_HormonizomSimilarity.SimilarityNormal>0.64)].sort_values(by='SimilarityNormal', ascending=False)


# HormonizomSimilarity_Hydroxychloroquine[['CID1','CID2']]
# HormonizomSimilarity_lopinavir[['CID1','CID2']]
# HormonizomSimilarity_Nitazoxanide[['CID1','CID2']]
# HormonizomSimilarity_disulfiram[['CID1','CID2']]

###### Candidate drug by high chemical and expression Similarity ########
Best_Similar_Hydroxychloroquine=TonimotoSimilarity_Hydroxychloroquine[TonimotoSimilarity_Hydroxychloroquine.CID2.isin(HormonizomSimilarity_Hydroxychloroquine.CID2)]
Best_Similar_disulfiram=TonimotoSimilarity_disulfiram[TonimotoSimilarity_disulfiram.CID2.isin(HormonizomSimilarity_disulfiram.CID2)]
Best_Similar_lopinavir=TonimotoSimilarity_lopinavir[TonimotoSimilarity_lopinavir.CID2.isin(HormonizomSimilarity_lopinavir.CID2)]
Best_Similar_Nitazoxanide=TonimotoSimilarity_Nitazoxanide[TonimotoSimilarity_Nitazoxanide.CID2.isin(HormonizomSimilarity_Nitazoxanide.CID2)]


###### Candidate drug by high chemical and expression Similarity  based each drug ########

TonimotoSimilarity_Hydroxychloroquine=Cluster5_tonimoto[(Cluster5_tonimoto.CID1==3652) & (Cluster5_tonimoto.Tonimoto>=0.6)].sort_values(by='Tonimoto', ascending=False)
HormonizomSimilarity_Hydroxychloroquine=Cluster5_HormonizomSimilarity[(Cluster5_HormonizomSimilarity.CID1==3652) & (Cluster5_HormonizomSimilarity.SimilarityNormal>=0.2) ].sort_values(by='SimilarityNormal', ascending=False)
Best_Similar_Hydroxychloroquine=pd.DataFrame(TonimotoSimilarity_Hydroxychloroquine[TonimotoSimilarity_Hydroxychloroquine.CID2.isin(HormonizomSimilarity_Hydroxychloroquine.CID2)])
Best_Similar_Hydroxychloroquine.insert(0,'DrugName','Hydroxychloroquine')

# Best_Similar_Hydroxychloroquine.CID2.to_csv('/Users/aghil/Desktop/eidd.csv',index=False)

TonimotoSimilarity_disulfiram=Cluster3_10_tonimoto[(Cluster3_10_tonimoto.CID1==3117) & (Cluster3_10_tonimoto.Tonimoto>=0.4)].sort_values(by='Tonimoto', ascending=False)
HormonizomSimilarity_disulfiram=Cluster3_10_HormonizomSimilarity[(Cluster3_10_HormonizomSimilarity.CID1==3117) & (Cluster3_10_HormonizomSimilarity.SimilarityNormal>=0.2)].sort_values(by='SimilarityNormal', ascending=False)
Best_Similar_disulfiram=pd.DataFrame(TonimotoSimilarity_disulfiram[TonimotoSimilarity_disulfiram.CID2.isin(HormonizomSimilarity_disulfiram.CID2)])
Best_Similar_disulfiram.insert(0,'DrugName','disulfiram')

TonimotoSimilarity_lopinavir=Cluster3_10_tonimoto[(Cluster3_10_tonimoto.CID1==92727) & (Cluster3_10_tonimoto.Tonimoto>0.65)].sort_values(by='Tonimoto', ascending=False)
HormonizomSimilarity_lopinavir=Cluster3_10_HormonizomSimilarity[(Cluster3_10_HormonizomSimilarity.CID1==92727)& (Cluster3_10_HormonizomSimilarity.SimilarityNormal>=0.2)].sort_values(by='SimilarityNormal', ascending=False)
Best_Similar_lopinavir=pd.DataFrame(TonimotoSimilarity_lopinavir[TonimotoSimilarity_lopinavir.CID2.isin(HormonizomSimilarity_lopinavir.CID2)])
Best_Similar_lopinavir.insert(0,'DrugName','lopinavir')

TonimotoSimilarity_Nitazoxanide=Cluster4_10_tonimoto[(Cluster4_10_tonimoto.CID1==41684) & (Cluster4_10_tonimoto.Tonimoto>0.55)].sort_values(by='Tonimoto', ascending=False)
HormonizomSimilarity_Nitazoxanide=Cluster4_10_HormonizomSimilarity[(Cluster4_10_HormonizomSimilarity.CID1==41684) & (Cluster4_10_HormonizomSimilarity.SimilarityNormal>0.2)].sort_values(by='SimilarityNormal', ascending=False)
Best_Similar_Nitazoxanide=pd.DataFrame(TonimotoSimilarity_Nitazoxanide[TonimotoSimilarity_Nitazoxanide.CID2.isin(HormonizomSimilarity_Nitazoxanide.CID2)])
Best_Similar_Nitazoxanide.insert(0,'DrugName','Nitazoxanide')

All_Candidate=Best_Similar_Hydroxychloroquine.append(Best_Similar_disulfiram)
All_Candidate=All_Candidate.append(Best_Similar_lopinavir)
All_Candidate=All_Candidate.append(Best_Similar_Nitazoxanide)


All_Candidate.to_csv('/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/All_New_Candidate_Drug_T_High_D_Low.csv',sep="\t",index=False)



# tonimoto[(tonimoto.CID1==3652) & (tonimoto.CID2==176870)].sort_values(by='Tonimoto', ascending=False)
# Similarity_Between_Hormonizom_CovidDrugCandid[(Similarity_Between_Hormonizom_CovidDrugCandid.CID1==3652) & (Similarity_Between_Hormonizom_CovidDrugCandid.CID2==176870)].sort_values(by='Similarity', ascending=False)

# tonimoto[(tonimoto.CID1==3117) & (tonimoto.CID2==176870)].sort_values(by='Tonimoto', ascending=False)
# Similarity_Between_Hormonizom_CovidDrugCandid[(Similarity_Between_Hormonizom_CovidDrugCandid.CID1==3117) & (Similarity_Between_Hormonizom_CovidDrugCandid.CID2==176870)].sort_values(by='Similarity', ascending=False)

# tonimoto[(tonimoto.CID1==41684) & (tonimoto.CID2==176870)].sort_values(by='Tonimoto', ascending=False)
# Similarity_Between_Hormonizom_CovidDrugCandid[(Similarity_Between_Hormonizom_CovidDrugCandid.CID1==41684) & (Similarity_Between_Hormonizom_CovidDrugCandid.CID2==176870)].sort_values(by='Similarity', ascending=False)

# tonimoto[(tonimoto.CID1==92727) & (tonimoto.CID2==176870)].sort_values(by='Tonimoto', ascending=False)
# Similarity_Between_Hormonizom_CovidDrugCandid[(Similarity_Between_Hormonizom_CovidDrugCandid.CID1==92727) & (Similarity_Between_Hormonizom_CovidDrugCandid.CID2==176870)].sort_values(by='Similarity', ascending=False)



#########  Evaluation of Clustering ##############
#Clustering Metrics
L4=pd.read_csv('/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/Out_L4.csv',delimiter="\t")
L4.loc[L4.PredictClass== 2,'PredictClass'] = 1
L4.loc[L4.PredictClass== 3,'PredictClass'] = 2
L4.loc[L4.PredictClass== 4,'PredictClass'] = 3
L4.loc[L4.PredictClass== 5,'PredictClass'] = 4
L4.loc[L4.PredictClass== 6,'PredictClass'] = 5
L4.loc[L4.PredictClass== 7,'PredictClass'] = 6
L4.loc[L4.PredictClass== 10,'PredictClass'] = 7
L4.loc[L4.PredictClass== 11,'PredictClass'] = 8
L4.loc[L4.PredictClass== 12,'PredictClass'] = 9
L4.loc[L4.PredictClass== 13,'PredictClass'] = 10
L4.loc[L4.PredictClass== 14,'PredictClass'] = 11
L4.loc[L4.PredictClass== 15,'PredictClass'] = 12
L4_WithoutNoClass=L4[L4.Class!='NoClass'].reset_index(drop=True)
L4_WithoutNoClass["Class"] = pd.to_numeric(L4_WithoutNoClass["Class"])

### Overlap Cluster by Class
L4_1=L4_WithoutNoClass[L4_WithoutNoClass.PredictClass==1].groupby(L4_WithoutNoClass['Class']).mean()
L4_2=L4_WithoutNoClass[L4_WithoutNoClass.PredictClass==2].groupby(L4_WithoutNoClass['Class']).mean()
L4_3=L4_WithoutNoClass[L4_WithoutNoClass.PredictClass==3].groupby(L4_WithoutNoClass['Class']).mean()
L4_4=L4_WithoutNoClass[L4_WithoutNoClass.PredictClass==4].groupby(L4_WithoutNoClass['Class']).mean()
L4_5=L4_WithoutNoClass[L4_WithoutNoClass.PredictClass==5].groupby(L4_WithoutNoClass['Class']).mean()
L4_6=L4_WithoutNoClass[L4_WithoutNoClass.PredictClass==6].groupby(L4_WithoutNoClass['Class']).mean()
L4_7=L4_WithoutNoClass[L4_WithoutNoClass.PredictClass==7].groupby(L4_WithoutNoClass['Class']).mean()
L4_8=L4_WithoutNoClass[L4_WithoutNoClass.PredictClass==8].groupby(L4_WithoutNoClass['Class']).mean()
L4_9=L4_WithoutNoClass[L4_WithoutNoClass.PredictClass==9].groupby(L4_WithoutNoClass['Class']).mean()
L4_10=L4_WithoutNoClass[L4_WithoutNoClass.PredictClass==10].groupby(L4_WithoutNoClass['Class']).mean()
L4_11=L4_WithoutNoClass[L4_WithoutNoClass.PredictClass==11].groupby(L4_WithoutNoClass['Class']).mean()
L4_12=L4_WithoutNoClass[L4_WithoutNoClass.PredictClass==12].groupby(L4_WithoutNoClass['Class']).mean()

# fig, axs = plt.subplots(3, 4)
# axs[0,0]=L4_2.plot.pie(y='Class', figsize=(15, 15), autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.2)
# axs[0,1]=L4_3.plot.pie(y='Class', figsize=(15, 15), autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.2)
# axs[0,2]=L4_4.plot.pie(y='Class', figsize=(15, 15), autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.2)
# axs[0,3]=L4_5.plot.pie(y='Class', figsize=(15, 15), autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.2)
# axs[1,0]=L4_6.plot.pie(y='Class', figsize=(15, 15), autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.2)
# axs[1,1]=L4_7.plot.pie(y='Class', figsize=(15, 15), autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.2)
# axs[1,2]=L4_10.plot.pie(y='Class', figsize=(15, 15), autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.2)
# axs[1,3]=L4_11.plot.pie(y='Class', figsize=(15, 15), autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.2)
# axs[2,0]=L4_12.plot.pie(y='Class', figsize=(15, 15), autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.2)
# axs[2,1]=L4_13.plot.pie(y='Class', figsize=(15, 15), autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.2)
# axs[2,2]=L4_14.plot.pie(y='Class', figsize=(15, 15), autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.2)
# axs[2,3]=L4_15.plot.pie(y='Class', figsize=(15, 15), autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.2)
# plt.show()


##### Metrics #######
Hormonizom=pd.read_csv("/Users/aghil/Desktop/uni/Tese 2/final list/Train_By Common Multimodal Data/only hormonizom/each Cell Line Dose And Time In One Row/All_Hormonizom_Col_Lower_VarQuantile0.75_Remove.csv",delimiter="\t")
CID=Hormonizom.CID
Class=Hormonizom.Class
Macc=pd.read_csv("/Users/aghil/Desktop/uni/Tese 2/final list/Train_By Common Multimodal Data/only hormonizom/each Cell Line Dose And Time In One Row/All_Macc_FP_Col_Var0_Remove.csv",delimiter="\t")
Hormonizom=Hormonizom.iloc[:,4:]
Macc=Macc.iloc[:,1:]
AllData=pd.concat([Macc,Hormonizom],axis=1)
AllData.insert(0,'CID',CID)
AllData.insert(0,'Class',Class)
AllData_WithoutNoClass=AllData[AllData.Class!='NoClass'].iloc[:,2:]
AllData_ByAllClass=AllData.iloc[:,2:]


metrics.adjusted_rand_score(np.array(L4_WithoutNoClass.Class,"int"),np.array(L4_WithoutNoClass.PredictClass,"int"))
metrics.normalized_mutual_info_score(np.array(L4_WithoutNoClass.Class,"int"),np.array(L4_WithoutNoClass.PredictClass,"int"))
metrics.adjusted_mutual_info_score(np.array(L4_WithoutNoClass.Class,"int"),np.array(L4_WithoutNoClass.PredictClass,"int"))
metrics.homogeneity_completeness_v_measure(np.array(L4_WithoutNoClass.Class,"int"),np.array(L4_WithoutNoClass.PredictClass,"int"))


metrics.calinski_harabasz_score(AllData.iloc[:,:],L4.PredictClass)
metrics.davies_bouldin_score(AllData.iloc[:,:],L4.PredictClass)
metrics.silhouette_score(AllData_WithoutNoClass,L4[L4.Class!='NoClass'].PredictClass)
metrics.silhouette_score(AllData_ByAllClass,L4.PredictClass.drop_duplicates())

# Kmean
kmeans = KMeans(n_clusters=11, random_state=0).fit(AllData_ByAllClass)
lable=kmeans.labels_
metrics.silhouette_score(AllData_ByAllClass,lable)

# PCA with ratio Macc and Hormonizom
pca = PCA(n_components=int(Macc.shape[1] * Macc.shape[1]/(Macc.shape[1]+Hormonizom.shape[1])))
pca.fit(Macc)
PcaMacc =pd.DataFrame( pca.transform(Macc))
pca = PCA(n_components=int(Hormonizom.shape[1] * Hormonizom.shape[1]/(Macc.shape[1]+Hormonizom.shape[1])))
pca.fit(Hormonizom)
PcaHormonizom =pd.DataFrame( pca.transform(Hormonizom))
AllPcaDataWithPca=pd.concat([PcaMacc,PcaHormonizom],axis=1)
kmeans = KMeans(n_clusters=11, random_state=0).fit(AllPcaDataWithPca)
lable=kmeans.labels_
metrics.silhouette_score(AllPcaDataWithPca,lable)

# PCA with L3 Data column Number
AllData=pd.concat([Macc,Hormonizom],axis=1)
pca = PCA(n_components=L3.shape[1])
pca.fit(AllData)
PcaAllData =pd.DataFrame( pca.transform(AllData))
kmeans = KMeans(n_clusters=7, random_state=0).fit(PcaAllData)
lable=kmeans.labels_
metrics.silhouette_score(PcaAllData,lable)


############ Kmeans Q_L3 ###############
L3=pd.read_csv('/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/Out_L3.csv',delimiter="\t")
L3=L3.iloc[:,2:]
kmeans = KMeans(n_clusters=7, random_state=0).fit(L3)
lable=kmeans.labels_
metrics.silhouette_score(L3,lable)

kmeans = KMeans(n_clusters=7, random_state=0).fit(Macc)
lable=kmeans.labels_
metrics.silhouette_score(Macc,lable)

kmeans = KMeans(n_clusters=7, random_state=0).fit(Hormonizom)
lable=kmeans.labels_
metrics.silhouette_score(Hormonizom,lable)



##SwissAdme Test For Cluster
All_Drug_Swissadme=pd.read_csv('/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/All_Drug_swissadme.csv',delimiter="\t")
All_Drug_Swissadme=All_Drug_Swissadme.dropna()

All_Drug_Swissadme=All_Drug_Swissadme[['Molecule','MW','#Heavy atoms','#Aromatic heavy atoms','Fraction Csp3','#Rotatable bonds','#H-bond acceptors','#H-bond donors','MR','TPSA','iLOGP','XLOGP3','WLOGP','MLOGP','Silicos-IT Log P','Synthetic Accessibility']]

All_Drug_Swissadme_L4_1=All_Drug_Swissadme[All_Drug_Swissadme.Molecule.isin(L4_WithoutNoClass[L4_WithoutNoClass.PredictClass==1]['CID'])]
All_Drug_Swissadme_L4_2=All_Drug_Swissadme[All_Drug_Swissadme.Molecule.isin(L4_WithoutNoClass[L4_WithoutNoClass.PredictClass==2]['CID'])]
All_Drug_Swissadme_L4_3=All_Drug_Swissadme[All_Drug_Swissadme.Molecule.isin(L4_WithoutNoClass[L4_WithoutNoClass.PredictClass==3]['CID'])]
All_Drug_Swissadme_L4_4=All_Drug_Swissadme[All_Drug_Swissadme.Molecule.isin(L4_WithoutNoClass[L4_WithoutNoClass.PredictClass==4]['CID'])]
All_Drug_Swissadme_L4_5=All_Drug_Swissadme[All_Drug_Swissadme.Molecule.isin(L4_WithoutNoClass[L4_WithoutNoClass.PredictClass==5]['CID'])]
All_Drug_Swissadme_L4_6=All_Drug_Swissadme[All_Drug_Swissadme.Molecule.isin(L4_WithoutNoClass[L4_WithoutNoClass.PredictClass==6]['CID'])]
All_Drug_Swissadme_L4_7=All_Drug_Swissadme[All_Drug_Swissadme.Molecule.isin(L4_WithoutNoClass[L4_WithoutNoClass.PredictClass==7]['CID'])]
All_Drug_Swissadme_L4_8=All_Drug_Swissadme[All_Drug_Swissadme.Molecule.isin(L4_WithoutNoClass[L4_WithoutNoClass.PredictClass==8]['CID'])]
All_Drug_Swissadme_L4_9=All_Drug_Swissadme[All_Drug_Swissadme.Molecule.isin(L4_WithoutNoClass[L4_WithoutNoClass.PredictClass==9]['CID'])]
All_Drug_Swissadme_L4_10=All_Drug_Swissadme[All_Drug_Swissadme.Molecule.isin(L4_WithoutNoClass[L4_WithoutNoClass.PredictClass==10]['CID'])]
All_Drug_Swissadme_L4_11=All_Drug_Swissadme[All_Drug_Swissadme.Molecule.isin(L4_WithoutNoClass[L4_WithoutNoClass.PredictClass==11]['CID'])]
All_Drug_Swissadme_L4_12=All_Drug_Swissadme[All_Drug_Swissadme.Molecule.isin(L4_WithoutNoClass[L4_WithoutNoClass.PredictClass==12]['CID'])]


sns.distplot(All_Drug_Swissadme_L4_1['#Heavy atoms'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 1"})
sns.distplot(All_Drug_Swissadme_L4_2['#Heavy atoms'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 2"})
sns.distplot(All_Drug_Swissadme_L4_3['#Heavy atoms'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 3"})
sns.distplot(All_Drug_Swissadme_L4_4['#Heavy atoms'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 4"})
sns.distplot(All_Drug_Swissadme_L4_5['#Heavy atoms'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 5"})
sns.distplot(All_Drug_Swissadme_L4_6['#Heavy atoms'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 6"})
sns.distplot(All_Drug_Swissadme_L4_7['#Heavy atoms'], kde=True,hist=False,kde_kws={"lw": 2, "label": "Cluster 7"})
sns.distplot(All_Drug_Swissadme_L4_8['#Heavy atoms'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 8"})
sns.distplot(All_Drug_Swissadme_L4_9['#Heavy atoms'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 9"})
sns.distplot(All_Drug_Swissadme_L4_10['#Heavy atoms'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 10"})
sns.distplot(All_Drug_Swissadme_L4_11['#Heavy atoms'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 11"})
sns.distplot(All_Drug_Swissadme_L4_12['#Heavy atoms'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 12"})


sns.distplot(All_Drug_Swissadme_L4_1['#Aromatic heavy atoms'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 1"})
sns.distplot(All_Drug_Swissadme_L4_2['#Aromatic heavy atoms'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 2"})
sns.distplot(All_Drug_Swissadme_L4_3['#Aromatic heavy atoms'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 3"})
sns.distplot(All_Drug_Swissadme_L4_4['#Aromatic heavy atoms'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 4"})
sns.distplot(All_Drug_Swissadme_L4_5['#Aromatic heavy atoms'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 5"})
sns.distplot(All_Drug_Swissadme_L4_6['#Aromatic heavy atoms'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 6"})
sns.distplot(All_Drug_Swissadme_L4_7['#Aromatic heavy atoms'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 7"})
sns.distplot(All_Drug_Swissadme_L4_8['#Aromatic heavy atoms'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 8"})
sns.distplot(All_Drug_Swissadme_L4_9['#Aromatic heavy atoms'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 9"})
sns.distplot(All_Drug_Swissadme_L4_10['#Aromatic heavy atoms'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 10"})
sns.distplot(All_Drug_Swissadme_L4_11['#Aromatic heavy atoms'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 11"})
sns.distplot(All_Drug_Swissadme_L4_12['#Aromatic heavy atoms'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 12"})


sns.distplot(All_Drug_Swissadme_L4_1['Fraction Csp3'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 1"})
sns.distplot(All_Drug_Swissadme_L4_2['Fraction Csp3'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 2"})
sns.distplot(All_Drug_Swissadme_L4_3['Fraction Csp3'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 3"})
sns.distplot(All_Drug_Swissadme_L4_4['Fraction Csp3'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 4"})
sns.distplot(All_Drug_Swissadme_L4_5['Fraction Csp3'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 5"})
sns.distplot(All_Drug_Swissadme_L4_6['Fraction Csp3'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 6"})
sns.distplot(All_Drug_Swissadme_L4_7['Fraction Csp3'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 7"})
sns.distplot(All_Drug_Swissadme_L4_8['Fraction Csp3'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 8"})
sns.distplot(All_Drug_Swissadme_L4_9['Fraction Csp3'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 9"})
sns.distplot(All_Drug_Swissadme_L4_10['Fraction Csp3'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 10"})
sns.distplot(All_Drug_Swissadme_L4_11['Fraction Csp3'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 11"})
sns.distplot(All_Drug_Swissadme_L4_12['Fraction Csp3'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 12"})


sns.distplot(All_Drug_Swissadme_L4_1['#Rotatable bonds'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 1"})
sns.distplot(All_Drug_Swissadme_L4_2['#Rotatable bonds'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 2"})
sns.distplot(All_Drug_Swissadme_L4_3['#Rotatable bonds'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 3"})
sns.distplot(All_Drug_Swissadme_L4_4['#Rotatable bonds'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 4"})
sns.distplot(All_Drug_Swissadme_L4_5['#Rotatable bonds'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 5"})
sns.distplot(All_Drug_Swissadme_L4_6['#Rotatable bonds'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 6"})
sns.distplot(All_Drug_Swissadme_L4_7['#Rotatable bonds'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 7"})
sns.distplot(All_Drug_Swissadme_L4_8['#Rotatable bonds'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 8"})
sns.distplot(All_Drug_Swissadme_L4_9['#Rotatable bonds'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 9"})
sns.distplot(All_Drug_Swissadme_L4_10['#Rotatable bonds'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 10"})
sns.distplot(All_Drug_Swissadme_L4_11['#Rotatable bonds'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 11"})
sns.distplot(All_Drug_Swissadme_L4_12['#Rotatable bonds'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 12"})


sns.distplot(All_Drug_Swissadme_L4_1['#H-bond acceptors'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 1"})
sns.distplot(All_Drug_Swissadme_L4_2['#H-bond acceptors'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 2"})
sns.distplot(All_Drug_Swissadme_L4_3['#H-bond acceptors'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 3"})
sns.distplot(All_Drug_Swissadme_L4_4['#H-bond acceptors'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 4"})
sns.distplot(All_Drug_Swissadme_L4_5['#H-bond acceptors'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 5"})
sns.distplot(All_Drug_Swissadme_L4_6['#H-bond acceptors'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 6"})
sns.distplot(All_Drug_Swissadme_L4_7['#H-bond acceptors'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 7"})
sns.distplot(All_Drug_Swissadme_L4_8['#H-bond acceptors'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 8"})
sns.distplot(All_Drug_Swissadme_L4_9['#H-bond acceptors'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 9"})
sns.distplot(All_Drug_Swissadme_L4_10['#H-bond acceptors'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 10"})
sns.distplot(All_Drug_Swissadme_L4_11['#H-bond acceptors'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 11"})
sns.distplot(All_Drug_Swissadme_L4_12['#H-bond acceptors'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 12"})


sns.distplot(All_Drug_Swissadme_L4_1['#H-bond donors'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 1"})
sns.distplot(All_Drug_Swissadme_L4_2['#H-bond donors'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 2"})
sns.distplot(All_Drug_Swissadme_L4_3['#H-bond donors'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 3"})
sns.distplot(All_Drug_Swissadme_L4_4['#H-bond donors'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 4"})
sns.distplot(All_Drug_Swissadme_L4_5['#H-bond donors'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 5"})
sns.distplot(All_Drug_Swissadme_L4_6['#H-bond donors'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 6"})
sns.distplot(All_Drug_Swissadme_L4_7['#H-bond donors'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 7"})
sns.distplot(All_Drug_Swissadme_L4_8['#H-bond donors'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 8"})
sns.distplot(All_Drug_Swissadme_L4_9['#H-bond donors'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 9"})
sns.distplot(All_Drug_Swissadme_L4_10['#H-bond donors'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 10"})
sns.distplot(All_Drug_Swissadme_L4_11['#H-bond donors'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 11"})
sns.distplot(All_Drug_Swissadme_L4_12['#H-bond donors'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 12"})


sns.distplot(All_Drug_Swissadme_L4_1['MR'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 1"})
sns.distplot(All_Drug_Swissadme_L4_2['MR'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 2"})
sns.distplot(All_Drug_Swissadme_L4_3['MR'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 3"})
sns.distplot(All_Drug_Swissadme_L4_4['MR'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 4"})
sns.distplot(All_Drug_Swissadme_L4_5['MR'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 5"})
sns.distplot(All_Drug_Swissadme_L4_6['MR'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 6"})
sns.distplot(All_Drug_Swissadme_L4_7['MR'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 7"})
sns.distplot(All_Drug_Swissadme_L4_8['MR'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 8"})
sns.distplot(All_Drug_Swissadme_L4_9['MR'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 9"})
sns.distplot(All_Drug_Swissadme_L4_10['MR'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 10"})
sns.distplot(All_Drug_Swissadme_L4_11['MR'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 11"})
sns.distplot(All_Drug_Swissadme_L4_12['MR'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 12"})


sns.distplot(All_Drug_Swissadme_L4_1['iLOGP'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 1"})
sns.distplot(All_Drug_Swissadme_L4_2['iLOGP'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 2"})
sns.distplot(All_Drug_Swissadme_L4_3['iLOGP'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 3"})
sns.distplot(All_Drug_Swissadme_L4_4['iLOGP'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 4"})
sns.distplot(All_Drug_Swissadme_L4_5['iLOGP'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 5"})
sns.distplot(All_Drug_Swissadme_L4_6['iLOGP'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 6"})
sns.distplot(All_Drug_Swissadme_L4_7['iLOGP'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 7"})
sns.distplot(All_Drug_Swissadme_L4_8['iLOGP'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 8"})
sns.distplot(All_Drug_Swissadme_L4_9['iLOGP'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 9"})
sns.distplot(All_Drug_Swissadme_L4_10['iLOGP'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 10"})
sns.distplot(All_Drug_Swissadme_L4_11['iLOGP'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 11"})
sns.distplot(All_Drug_Swissadme_L4_12['iLOGP'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 12"})


sns.distplot(All_Drug_Swissadme_L4_1['TPSA'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 1"})
sns.distplot(All_Drug_Swissadme_L4_2['TPSA'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 2"})
sns.distplot(All_Drug_Swissadme_L4_3['TPSA'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 3"})
sns.distplot(All_Drug_Swissadme_L4_4['TPSA'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 4"})
sns.distplot(All_Drug_Swissadme_L4_5['TPSA'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 5"})
sns.distplot(All_Drug_Swissadme_L4_6['TPSA'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 6"})
sns.distplot(All_Drug_Swissadme_L4_7['TPSA'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 7"})
sns.distplot(All_Drug_Swissadme_L4_8['TPSA'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 8"})
sns.distplot(All_Drug_Swissadme_L4_9['TPSA'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 9"})
sns.distplot(All_Drug_Swissadme_L4_10['TPSA'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 10"})
sns.distplot(All_Drug_Swissadme_L4_11['TPSA'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 11"})
sns.distplot(All_Drug_Swissadme_L4_12['TPSA'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 12"})


sns.distplot(All_Drug_Swissadme_L4_1['XLOGP3'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 1"})
sns.distplot(All_Drug_Swissadme_L4_2['XLOGP3'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 2"})
sns.distplot(All_Drug_Swissadme_L4_3['XLOGP3'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 3"})
sns.distplot(All_Drug_Swissadme_L4_4['XLOGP3'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 4"})
sns.distplot(All_Drug_Swissadme_L4_5['XLOGP3'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 5"})
sns.distplot(All_Drug_Swissadme_L4_6['XLOGP3'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 6"})
sns.distplot(All_Drug_Swissadme_L4_7['XLOGP3'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 7"})
sns.distplot(All_Drug_Swissadme_L4_8['XLOGP3'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 8"})
sns.distplot(All_Drug_Swissadme_L4_9['XLOGP3'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 9"})
sns.distplot(All_Drug_Swissadme_L4_10['XLOGP3'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 10"})
sns.distplot(All_Drug_Swissadme_L4_11['XLOGP3'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 11"})
sns.distplot(All_Drug_Swissadme_L4_12['XLOGP3'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 12"})


sns.distplot(All_Drug_Swissadme_L4_1['WLOGP'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 1"})
sns.distplot(All_Drug_Swissadme_L4_2['WLOGP'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 2"})
sns.distplot(All_Drug_Swissadme_L4_3['WLOGP'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 3"})
sns.distplot(All_Drug_Swissadme_L4_4['WLOGP'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 4"})
sns.distplot(All_Drug_Swissadme_L4_5['WLOGP'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 5"})
sns.distplot(All_Drug_Swissadme_L4_6['WLOGP'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 6"})
sns.distplot(All_Drug_Swissadme_L4_7['WLOGP'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 7"})
sns.distplot(All_Drug_Swissadme_L4_8['WLOGP'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 8"})
sns.distplot(All_Drug_Swissadme_L4_9['WLOGP'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 9"})
sns.distplot(All_Drug_Swissadme_L4_10['WLOGP'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 10"})
sns.distplot(All_Drug_Swissadme_L4_11['WLOGP'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 11"})
sns.distplot(All_Drug_Swissadme_L4_12['WLOGP'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 12"})


sns.distplot(All_Drug_Swissadme_L4_1['MLOGP'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 1"})
sns.distplot(All_Drug_Swissadme_L4_2['MLOGP'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 2"})
sns.distplot(All_Drug_Swissadme_L4_3['MLOGP'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 3"})
sns.distplot(All_Drug_Swissadme_L4_4['MLOGP'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 4"})
sns.distplot(All_Drug_Swissadme_L4_5['MLOGP'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 5"})
sns.distplot(All_Drug_Swissadme_L4_6['MLOGP'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 6"})
sns.distplot(All_Drug_Swissadme_L4_7['MLOGP'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 7"})
sns.distplot(All_Drug_Swissadme_L4_8['MLOGP'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 8"})
sns.distplot(All_Drug_Swissadme_L4_9['MLOGP'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 9"})
sns.distplot(All_Drug_Swissadme_L4_10['MLOGP'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 10"})
sns.distplot(All_Drug_Swissadme_L4_11['MLOGP'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 11"})
sns.distplot(All_Drug_Swissadme_L4_12['MLOGP'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 12"})


sns.distplot(All_Drug_Swissadme_L4_1['Silicos-IT Log P'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 1"})
sns.distplot(All_Drug_Swissadme_L4_2['Silicos-IT Log P'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 2"})
sns.distplot(All_Drug_Swissadme_L4_3['Silicos-IT Log P'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 3"})
sns.distplot(All_Drug_Swissadme_L4_4['Silicos-IT Log P'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 4"})
sns.distplot(All_Drug_Swissadme_L4_5['Silicos-IT Log P'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 5"})
sns.distplot(All_Drug_Swissadme_L4_6['Silicos-IT Log P'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 6"})
sns.distplot(All_Drug_Swissadme_L4_7['Silicos-IT Log P'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 7"})
sns.distplot(All_Drug_Swissadme_L4_8['Silicos-IT Log P'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 8"})
sns.distplot(All_Drug_Swissadme_L4_9['Silicos-IT Log P'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 9"})
sns.distplot(All_Drug_Swissadme_L4_10['Silicos-IT Log P'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 10"})
sns.distplot(All_Drug_Swissadme_L4_11['Silicos-IT Log P'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 11"})
sns.distplot(All_Drug_Swissadme_L4_12['Silicos-IT Log P'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 12"})


sns.distplot(All_Drug_Swissadme_L4_1['Synthetic Accessibility'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 1"})
sns.distplot(All_Drug_Swissadme_L4_2['Synthetic Accessibility'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 2"})
sns.distplot(All_Drug_Swissadme_L4_3['Synthetic Accessibility'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 3"})
sns.distplot(All_Drug_Swissadme_L4_4['Synthetic Accessibility'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 4"})
sns.distplot(All_Drug_Swissadme_L4_5['Synthetic Accessibility'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 5"})
sns.distplot(All_Drug_Swissadme_L4_6['Synthetic Accessibility'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 6"})
sns.distplot(All_Drug_Swissadme_L4_7['Synthetic Accessibility'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 7"})
sns.distplot(All_Drug_Swissadme_L4_8['Synthetic Accessibility'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 8"})
sns.distplot(All_Drug_Swissadme_L4_9['Synthetic Accessibility'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 9"})
sns.distplot(All_Drug_Swissadme_L4_10['Synthetic Accessibility'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 10"})
sns.distplot(All_Drug_Swissadme_L4_11['Synthetic Accessibility'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 11"})
sns.distplot(All_Drug_Swissadme_L4_12['Synthetic Accessibility'], kde=True,hist=False,kde_kws={ "lw": 2, "label": "Cluster 12"})











# All_Drug_Swissadme_L4_2_CYP1A2_inhibitor=All_Drug_Swissadme_L4_2['CYP1A2 inhibitor'].groupby(All_Drug_Swissadme_L4_2['CYP1A2 inhibitor']).count()
# All_Drug_Swissadme_L4_3_CYP1A2_inhibitor=All_Drug_Swissadme_L4_3['CYP1A2 inhibitor'].groupby(All_Drug_Swissadme_L4_3['CYP1A2 inhibitor']).count()
# All_Drug_Swissadme_L4_4_CYP1A2_inhibitor=All_Drug_Swissadme_L4_4['CYP1A2 inhibitor'].groupby(All_Drug_Swissadme_L4_4['CYP1A2 inhibitor']).count()
# All_Drug_Swissadme_L4_5_CYP1A2_inhibitor=All_Drug_Swissadme_L4_5['CYP1A2 inhibitor'].groupby(All_Drug_Swissadme_L4_5['CYP1A2 inhibitor']).count()
# All_Drug_Swissadme_L4_6_CYP1A2_inhibitor=All_Drug_Swissadme_L4_6['CYP1A2 inhibitor'].groupby(All_Drug_Swissadme_L4_6['CYP1A2 inhibitor']).count()
# All_Drug_Swissadme_L4_7_CYP1A2_inhibitor=All_Drug_Swissadme_L4_7['CYP1A2 inhibitor'].groupby(All_Drug_Swissadme_L4_7['CYP1A2 inhibitor']).count()
# All_Drug_Swissadme_L4_10_CYP1A2_inhibitor=All_Drug_Swissadme_L4_10['CYP1A2 inhibitor'].groupby(All_Drug_Swissadme_L4_10['CYP1A2 inhibitor']).count()
# All_Drug_Swissadme_L4_11_CYP1A2_inhibitor=All_Drug_Swissadme_L4_11['CYP1A2 inhibitor'].groupby(All_Drug_Swissadme_L4_11['CYP1A2 inhibitor']).count()
# All_Drug_Swissadme_L4_12_CYP1A2_inhibitor=All_Drug_Swissadme_L4_11['CYP1A2 inhibitor'].groupby(All_Drug_Swissadme_L4_12['CYP1A2 inhibitor']).count()
# All_Drug_Swissadme_L4_13_CYP1A2_inhibitor=All_Drug_Swissadme_L4_13['CYP1A2 inhibitor'].groupby(All_Drug_Swissadme_L4_13['CYP1A2 inhibitor']).count()
# All_Drug_Swissadme_L4_14_CYP1A2_inhibitor=All_Drug_Swissadme_L4_14['CYP1A2 inhibitor'].groupby(All_Drug_Swissadme_L4_14['CYP1A2 inhibitor']).count()
# All_Drug_Swissadme_L4_15_CYP1A2_inhibitor=All_Drug_Swissadme_L4_15['CYP1A2 inhibitor'].groupby(All_Drug_Swissadme_L4_15['CYP1A2 inhibitor']).count()




# fig, axs = plt.subplots(3, 4)
# axs[0,0]=All_Drug_Swissadme_L4_2_CYP1A2_inhibitor.plot.pie(y='CYP1A2 inhibitor', figsize=(15, 15), autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.2)
# axs[0,1]=All_Drug_Swissadme_L4_3_CYP1A2_inhibitor.plot.pie(y='CYP1A2 inhibitor', figsize=(15, 15), autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.2)
# axs[0,2]=All_Drug_Swissadme_L4_4_CYP1A2_inhibitor.plot.pie(y='CYP1A2 inhibitor', figsize=(15, 15), autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.2)
# axs[0,3]=All_Drug_Swissadme_L4_5_CYP1A2_inhibitor.plot.pie(y='CYP1A2 inhibitor', figsize=(15, 15), autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.2)
# axs[1,0]=All_Drug_Swissadme_L4_6_CYP1A2_inhibitor.plot.pie(y='CYP1A2 inhibitor', figsize=(15, 15), autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.2)
# axs[1,1]=All_Drug_Swissadme_L4_7_CYP1A2_inhibitor.plot.pie(y='CYP1A2 inhibitor', figsize=(15, 15), autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.2)
# axs[1,2]=All_Drug_Swissadme_L4_10_CYP1A2_inhibitor.plot.pie(y='CYP1A2 inhibitor', figsize=(15, 15), autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.2)
# axs[1,3]=All_Drug_Swissadme_L4_11_CYP1A2_inhibitor.plot.pie(y='CYP1A2 inhibitor', figsize=(15, 15), autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.2)
# axs[2,0]=All_Drug_Swissadme_L4_12_CYP1A2_inhibitor.plot.pie(y='CYP1A2 inhibitor', figsize=(15, 15), autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.2)
# axs[2,1]=All_Drug_Swissadme_L4_13_CYP1A2_inhibitor.plot.pie(y='CYP1A2 inhibitor', figsize=(15, 15), autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.2)
# axs[2,2]=All_Drug_Swissadme_L4_14_CYP1A2_inhibitor.plot.pie(y='CYP1A2 inhibitor', figsize=(15, 15), autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.2)
# axs[2,3]=All_Drug_Swissadme_L4_15_CYP1A2_inhibitor.plot.pie(y='CYP1A2 inhibitor', figsize=(15, 15), autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.2)
# plt.show()

####### Mean of SwissADME for even cluster  without Merge #######
L4=pd.read_csv('/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/Out_L4.csv',delimiter="\t")
L4.loc[L4.PredictClass== 2,'PredictClass'] = 1
L4.loc[L4.PredictClass== 3,'PredictClass'] = 2
L4.loc[L4.PredictClass== 4,'PredictClass'] = 3
L4.loc[L4.PredictClass== 5,'PredictClass'] = 4
L4.loc[L4.PredictClass== 6,'PredictClass'] = 5
L4.loc[L4.PredictClass== 7,'PredictClass'] = 6
L4.loc[L4.PredictClass== 10,'PredictClass'] = 7
L4.loc[L4.PredictClass== 11,'PredictClass'] = 8
L4.loc[L4.PredictClass== 12,'PredictClass'] = 9
L4.loc[L4.PredictClass== 13,'PredictClass'] = 10
L4.loc[L4.PredictClass== 14,'PredictClass'] = 11
L4.loc[L4.PredictClass== 15,'PredictClass'] = 12
L4_WithoutNoClass=L4[L4.Class!='NoClass'].reset_index(drop=True)
L4_WithoutNoClass["Class"] = pd.to_numeric(L4_WithoutNoClass["Class"])
L4_WithoutNoClass_Merge=L4_WithoutNoClass

All_Drug_Swissadme_L4_1=All_Drug_Swissadme[All_Drug_Swissadme.Molecule.isin(L4_WithoutNoClass_Merge[L4_WithoutNoClass_Merge.PredictClass==1]['CID'])]
All_Drug_Swissadme_L4_2=All_Drug_Swissadme[All_Drug_Swissadme.Molecule.isin(L4_WithoutNoClass_Merge[L4_WithoutNoClass_Merge.PredictClass==2]['CID'])]
All_Drug_Swissadme_L4_3=All_Drug_Swissadme[All_Drug_Swissadme.Molecule.isin(L4_WithoutNoClass_Merge[L4_WithoutNoClass_Merge.PredictClass==3]['CID'])]
All_Drug_Swissadme_L4_4=All_Drug_Swissadme[All_Drug_Swissadme.Molecule.isin(L4_WithoutNoClass_Merge[L4_WithoutNoClass_Merge.PredictClass==4]['CID'])]
All_Drug_Swissadme_L4_5=All_Drug_Swissadme[All_Drug_Swissadme.Molecule.isin(L4_WithoutNoClass_Merge[L4_WithoutNoClass_Merge.PredictClass==5]['CID'])]
All_Drug_Swissadme_L4_6=All_Drug_Swissadme[All_Drug_Swissadme.Molecule.isin(L4_WithoutNoClass_Merge[L4_WithoutNoClass_Merge.PredictClass==6]['CID'])]
All_Drug_Swissadme_L4_7=All_Drug_Swissadme[All_Drug_Swissadme.Molecule.isin(L4_WithoutNoClass_Merge[L4_WithoutNoClass_Merge.PredictClass==7]['CID'])]
All_Drug_Swissadme_L4_8=All_Drug_Swissadme[All_Drug_Swissadme.Molecule.isin(L4_WithoutNoClass_Merge[L4_WithoutNoClass_Merge.PredictClass==8]['CID'])]
All_Drug_Swissadme_L4_9=All_Drug_Swissadme[All_Drug_Swissadme.Molecule.isin(L4_WithoutNoClass_Merge[L4_WithoutNoClass_Merge.PredictClass==9]['CID'])]
All_Drug_Swissadme_L4_10=All_Drug_Swissadme[All_Drug_Swissadme.Molecule.isin(L4_WithoutNoClass_Merge[L4_WithoutNoClass_Merge.PredictClass==10]['CID'])]
All_Drug_Swissadme_L4_11=All_Drug_Swissadme[All_Drug_Swissadme.Molecule.isin(L4_WithoutNoClass_Merge[L4_WithoutNoClass_Merge.PredictClass==11]['CID'])]
All_Drug_Swissadme_L4_12=All_Drug_Swissadme[All_Drug_Swissadme.Molecule.isin(L4_WithoutNoClass_Merge[L4_WithoutNoClass_Merge.PredictClass==12]['CID'])]

All_Drug_Swissadme_L4_1.insert(0,'Cluster',1)
All_Drug_Swissadme_L4_2.insert(0,'Cluster',2)
All_Drug_Swissadme_L4_3.insert(0,'Cluster',3)
All_Drug_Swissadme_L4_4.insert(0,'Cluster',4)
All_Drug_Swissadme_L4_5.insert(0,'Cluster',5)
All_Drug_Swissadme_L4_6.insert(0,'Cluster',6)
All_Drug_Swissadme_L4_7.insert(0,'Cluster',7)
All_Drug_Swissadme_L4_8.insert(0,'Cluster',8)
All_Drug_Swissadme_L4_9.insert(0,'Cluster',9)
All_Drug_Swissadme_L4_10.insert(0,'Cluster',10)
All_Drug_Swissadme_L4_11.insert(0,'Cluster',11)
All_Drug_Swissadme_L4_12.insert(0,'Cluster',12)

All_Drug_Swissadme_ByClass=All_Drug_Swissadme_L4_1.append(All_Drug_Swissadme_L4_2)
All_Drug_Swissadme_ByClass=All_Drug_Swissadme_ByClass.append(All_Drug_Swissadme_L4_3)
All_Drug_Swissadme_ByClass=All_Drug_Swissadme_ByClass.append(All_Drug_Swissadme_L4_4)
All_Drug_Swissadme_ByClass=All_Drug_Swissadme_ByClass.append(All_Drug_Swissadme_L4_5)
All_Drug_Swissadme_ByClass=All_Drug_Swissadme_ByClass.append(All_Drug_Swissadme_L4_6)
All_Drug_Swissadme_ByClass=All_Drug_Swissadme_ByClass.append(All_Drug_Swissadme_L4_7)
All_Drug_Swissadme_ByClass=All_Drug_Swissadme_ByClass.append(All_Drug_Swissadme_L4_8)
All_Drug_Swissadme_ByClass=All_Drug_Swissadme_ByClass.append(All_Drug_Swissadme_L4_9)
All_Drug_Swissadme_ByClass=All_Drug_Swissadme_ByClass.append(All_Drug_Swissadme_L4_10)
All_Drug_Swissadme_ByClass=All_Drug_Swissadme_ByClass.append(All_Drug_Swissadme_L4_11)
All_Drug_Swissadme_ByClass=All_Drug_Swissadme_ByClass.append(All_Drug_Swissadme_L4_12)

## All Cluster All PhysicoChemical Summary
PhysicoArr=['MW','#Heavy atoms','#Aromatic heavy atoms','Fraction Csp3','#Rotatable bonds','#H-bond acceptors','#H-bond donors','MR','TPSA','iLOGP','XLOGP3','WLOGP','MLOGP','Silicos-IT Log P','Synthetic Accessibility']
# AllCluster_PhysicoChemical=pd.DataFrame(columns=['Variable', 'N', 'Mean', 'SD', 'SE', '95% Conf.', 'Interval'])
pd.DataFrame(rp.summary_cont(All_Drug_Swissadme_ByClass[PhysicoArr[0]].groupby(All_Drug_Swissadme_ByClass['Cluster']))).to_csv('/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/Cluster Overlap By Class/all cluster/'+'PysicoChemical_'+str(PhysicoArr[0])+'_Summary_All_Cluster.csv',sep='\t',index=False)
pd.DataFrame(rp.summary_cont(All_Drug_Swissadme_ByClass[PhysicoArr[1]].groupby(All_Drug_Swissadme_ByClass['Cluster']))).to_csv('/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/Cluster Overlap By Class/all cluster/'+'PysicoChemical_'+str(PhysicoArr[1])+'_Summary_All_Cluster.csv',sep='\t',index=False)
pd.DataFrame(rp.summary_cont(All_Drug_Swissadme_ByClass[PhysicoArr[2]].groupby(All_Drug_Swissadme_ByClass['Cluster']))).to_csv('/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/Cluster Overlap By Class/all cluster/'+'PysicoChemical_'+str(PhysicoArr[2])+'_Summary_All_Cluster.csv',sep='\t',index=False)
pd.DataFrame(rp.summary_cont(All_Drug_Swissadme_ByClass[PhysicoArr[3]].groupby(All_Drug_Swissadme_ByClass['Cluster']))).to_csv('/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/Cluster Overlap By Class/all cluster/'+'PysicoChemical_'+str(PhysicoArr[3])+'_Summary_All_Cluster.csv',sep='\t',index=False)
pd.DataFrame(rp.summary_cont(All_Drug_Swissadme_ByClass[PhysicoArr[4]].groupby(All_Drug_Swissadme_ByClass['Cluster']))).to_csv('/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/Cluster Overlap By Class/all cluster/'+'PysicoChemical_'+str(PhysicoArr[4])+'_Summary_All_Cluster.csv',sep='\t',index=False)
pd.DataFrame(rp.summary_cont(All_Drug_Swissadme_ByClass[PhysicoArr[5]].groupby(All_Drug_Swissadme_ByClass['Cluster']))).to_csv('/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/Cluster Overlap By Class/all cluster/'+'PysicoChemical_'+str(PhysicoArr[5])+'_Summary_All_Cluster.csv',sep='\t',index=False)
pd.DataFrame(rp.summary_cont(All_Drug_Swissadme_ByClass[PhysicoArr[6]].groupby(All_Drug_Swissadme_ByClass['Cluster']))).to_csv('/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/Cluster Overlap By Class/all cluster/'+'PysicoChemical_'+str(PhysicoArr[6])+'_Summary_All_Cluster.csv',sep='\t',index=False)
pd.DataFrame(rp.summary_cont(All_Drug_Swissadme_ByClass[PhysicoArr[7]].groupby(All_Drug_Swissadme_ByClass['Cluster']))).to_csv('/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/Cluster Overlap By Class/all cluster/'+'PysicoChemical_'+str(PhysicoArr[7])+'_Summary_All_Cluster.csv',sep='\t',index=False)
pd.DataFrame(rp.summary_cont(All_Drug_Swissadme_ByClass[PhysicoArr[8]].groupby(All_Drug_Swissadme_ByClass['Cluster']))).to_csv('/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/Cluster Overlap By Class/all cluster/'+'PysicoChemical_'+str(PhysicoArr[8])+'_Summary_All_Cluster.csv',sep='\t',index=False)
pd.DataFrame(rp.summary_cont(All_Drug_Swissadme_ByClass[PhysicoArr[9]].groupby(All_Drug_Swissadme_ByClass['Cluster']))).to_csv('/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/Cluster Overlap By Class/all cluster/'+'PysicoChemical_'+str(PhysicoArr[9])+'_Summary_All_Cluster.csv',sep='\t',index=False)
pd.DataFrame(rp.summary_cont(All_Drug_Swissadme_ByClass[PhysicoArr[10]].groupby(All_Drug_Swissadme_ByClass['Cluster']))).to_csv('/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/Cluster Overlap By Class/all cluster/'+'PysicoChemical_'+str(PhysicoArr[10])+'_Summary_All_Cluster.csv',sep='\t',index=False)
pd.DataFrame(rp.summary_cont(All_Drug_Swissadme_ByClass[PhysicoArr[11]].groupby(All_Drug_Swissadme_ByClass['Cluster']))).to_csv('/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/Cluster Overlap By Class/all cluster/'+'PysicoChemical_'+str(PhysicoArr[11])+'_Summary_All_Cluster.csv',sep='\t',index=False)
pd.DataFrame(rp.summary_cont(All_Drug_Swissadme_ByClass[PhysicoArr[12]].groupby(All_Drug_Swissadme_ByClass['Cluster']))).to_csv('/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/Cluster Overlap By Class/all cluster/'+'PysicoChemical_'+str(PhysicoArr[12])+'_Summary_All_Cluster.csv',sep='\t',index=False)


sns.boxplot(data=All_Drug_Swissadme_ByClass,x='Cluster',y='MW')
sns.boxplot(data=All_Drug_Swissadme_ByClass,x='Cluster',y='#Heavy atoms')
sns.boxplot(data=All_Drug_Swissadme_ByClass,x='Cluster',y='#Aromatic heavy atoms')
sns.boxplot(data=All_Drug_Swissadme_ByClass,x='Cluster',y='Fraction Csp3')
sns.boxplot(data=All_Drug_Swissadme_ByClass,x='Cluster',y='#Rotatable bonds')
sns.boxplot(data=All_Drug_Swissadme_ByClass,x='Cluster',y='#H-bond acceptors')
sns.boxplot(data=All_Drug_Swissadme_ByClass,x='Cluster',y='#H-bond donors')
sns.boxplot(data=All_Drug_Swissadme_ByClass,x='Cluster',y='MR')
sns.boxplot(data=All_Drug_Swissadme_ByClass,x='Cluster',y='TPSA')
sns.boxplot(data=All_Drug_Swissadme_ByClass,x='Cluster',y='iLOGP')
sns.boxplot(data=All_Drug_Swissadme_ByClass,x='Cluster',y='XLOGP3')
sns.boxplot(data=All_Drug_Swissadme_ByClass,x='Cluster',y='WLOGP')
sns.boxplot(data=All_Drug_Swissadme_ByClass,x='Cluster',y='MLOGP')
sns.boxplot(data=All_Drug_Swissadme_ByClass,x='Cluster',y='Silicos-IT Log P')
sns.boxplot(data=All_Drug_Swissadme_ByClass,x='Cluster',y='Synthetic Accessibility')


### Anova Test Physicochemical all cluster
SwissADMEAnova=pd.DataFrame(columns=['Feature','FStatistic','Pvalue'])
for i in range(13):
   j=len(SwissADMEAnova)
   result_Statistic=stats.f_oneway(All_Drug_Swissadme_ByClass[All_Drug_Swissadme_ByClass.Cluster==1][PhysicoArr[i]]
                                                   ,All_Drug_Swissadme_ByClass[All_Drug_Swissadme_ByClass.Cluster==2][PhysicoArr[i]]
                                                   ,All_Drug_Swissadme_ByClass[All_Drug_Swissadme_ByClass.Cluster==3][PhysicoArr[i]]
                                                   ,All_Drug_Swissadme_ByClass[All_Drug_Swissadme_ByClass.Cluster==4][PhysicoArr[i]]
                                                   ,All_Drug_Swissadme_ByClass[All_Drug_Swissadme_ByClass.Cluster==5][PhysicoArr[i]]
                                                   ,All_Drug_Swissadme_ByClass[All_Drug_Swissadme_ByClass.Cluster==6][PhysicoArr[i]]
                                                   ,All_Drug_Swissadme_ByClass[All_Drug_Swissadme_ByClass.Cluster==7][PhysicoArr[i]]
                                                   ,All_Drug_Swissadme_ByClass[All_Drug_Swissadme_ByClass.Cluster==8][PhysicoArr[i]]
                                                   ,All_Drug_Swissadme_ByClass[All_Drug_Swissadme_ByClass.Cluster==9][PhysicoArr[i]]
                                                   ,All_Drug_Swissadme_ByClass[All_Drug_Swissadme_ByClass.Cluster==10][PhysicoArr[i]]
                                                   ,All_Drug_Swissadme_ByClass[All_Drug_Swissadme_ByClass.Cluster==11][PhysicoArr[i]]
                                                   ,All_Drug_Swissadme_ByClass[All_Drug_Swissadme_ByClass.Cluster==12][PhysicoArr[i]])[0]
    
   result_PValue=stats.f_oneway(All_Drug_Swissadme_ByClass[All_Drug_Swissadme_ByClass.Cluster==1][PhysicoArr[i]]
                                                   ,All_Drug_Swissadme_ByClass[All_Drug_Swissadme_ByClass.Cluster==2][PhysicoArr[i]]
                                                   ,All_Drug_Swissadme_ByClass[All_Drug_Swissadme_ByClass.Cluster==3][PhysicoArr[i]]
                                                   ,All_Drug_Swissadme_ByClass[All_Drug_Swissadme_ByClass.Cluster==4][PhysicoArr[i]]
                                                   ,All_Drug_Swissadme_ByClass[All_Drug_Swissadme_ByClass.Cluster==5][PhysicoArr[i]]
                                                   ,All_Drug_Swissadme_ByClass[All_Drug_Swissadme_ByClass.Cluster==6][PhysicoArr[i]]
                                                   ,All_Drug_Swissadme_ByClass[All_Drug_Swissadme_ByClass.Cluster==7][PhysicoArr[i]]
                                                   ,All_Drug_Swissadme_ByClass[All_Drug_Swissadme_ByClass.Cluster==8][PhysicoArr[i]]
                                                   ,All_Drug_Swissadme_ByClass[All_Drug_Swissadme_ByClass.Cluster==9][PhysicoArr[i]]
                                                   ,All_Drug_Swissadme_ByClass[All_Drug_Swissadme_ByClass.Cluster==10][PhysicoArr[i]]
                                                   ,All_Drug_Swissadme_ByClass[All_Drug_Swissadme_ByClass.Cluster==11][PhysicoArr[i]]
                                                   ,All_Drug_Swissadme_ByClass[All_Drug_Swissadme_ByClass.Cluster==12][PhysicoArr[i]])[1]
   Concat = PhysicoArr[i] + ' - ' + str(result_Statistic) + ' - ' + str(result_PValue)
   SwissADMEAnova.at[j,'Feature']=PhysicoArr[i]
   SwissADMEAnova.at[j,'FStatistic']=np.round(result_Statistic,4)
   SwissADMEAnova.at[j,'Pvalue']=np.round(result_PValue,4)

SwissADMEAnova.to_csv('/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/Cluster Overlap By Class/all cluster/Anova_Swissadme_AllCluster.csv',sep="\t",index=False)

### Anova Test MACC
Macc.insert(0,'Cluster',L4.PredictClass)

MaccAnova=pd.DataFrame(columns=['MaccFeature','FStatistic','Pvalue'])
for Col in Macc.columns[2:]:
   j=len(MaccAnova)
   result_Statistic=stats.f_oneway(Macc[Macc.Cluster==1][Col]
                                    ,Macc[Macc.Cluster==2][Col]
                                    ,Macc[Macc.Cluster==3][Col]
                                    ,Macc[Macc.Cluster==4][Col]
                                    ,Macc[Macc.Cluster==5][Col]
                                    ,Macc[Macc.Cluster==6][Col]
                                    ,Macc[Macc.Cluster==7][Col]
                                    ,Macc[Macc.Cluster==8][Col]
                                    ,Macc[Macc.Cluster==9][Col]
                                    ,Macc[Macc.Cluster==10][Col]
                                    ,Macc[Macc.Cluster==11][Col]
                                    ,Macc[Macc.Cluster==12][Col])[0]
    
   result_PValue=stats.f_oneway(Macc[Macc.Cluster==1][Col]
                                    ,Macc[Macc.Cluster==2][Col]
                                    ,Macc[Macc.Cluster==3][Col]
                                    ,Macc[Macc.Cluster==4][Col]
                                    ,Macc[Macc.Cluster==5][Col]
                                    ,Macc[Macc.Cluster==6][Col]
                                    ,Macc[Macc.Cluster==7][Col]
                                    ,Macc[Macc.Cluster==8][Col]
                                    ,Macc[Macc.Cluster==9][Col]
                                    ,Macc[Macc.Cluster==10][Col]
                                    ,Macc[Macc.Cluster==11][Col]
                                    ,Macc[Macc.Cluster==12][Col])[1]
   if result_PValue<=0.05:
       MaccAnova.at[j,'MaccFeature']=Col
       MaccAnova.at[j,'FStatistic']=np.round(result_Statistic,4)
       MaccAnova.at[j,'Pvalue']=np.round(result_PValue,4)

MaccAnova.to_csv('/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/Cluster Overlap By Class/all cluster/Anova_Macc_AllCluster.csv',sep="\t",index=False)


### Anova Test Hormonizom
count=0
Hormonizom.insert(4,'Cluster',L4.PredictClass)
HormonizomAnova=pd.DataFrame(columns=['HormonizomFeature','FStatistic','Pvalue'])
for Col in Hormonizom.columns[5:]:
   count=count+1
   print(count)
   j=len(HormonizomAnova)
   result_Statistic=stats.f_oneway(Hormonizom[Hormonizom.Cluster==1][Col]
                                    ,Hormonizom[Hormonizom.Cluster==2][Col]
                                    ,Hormonizom[Hormonizom.Cluster==3][Col]
                                    ,Hormonizom[Hormonizom.Cluster==4][Col]
                                    ,Hormonizom[Hormonizom.Cluster==5][Col]
                                    ,Hormonizom[Hormonizom.Cluster==6][Col]
                                    ,Hormonizom[Hormonizom.Cluster==7][Col]
                                    ,Hormonizom[Hormonizom.Cluster==8][Col]
                                    ,Hormonizom[Hormonizom.Cluster==9][Col]
                                    ,Hormonizom[Hormonizom.Cluster==10][Col]
                                    ,Hormonizom[Hormonizom.Cluster==11][Col]
                                    ,Hormonizom[Hormonizom.Cluster==12][Col])[0]
    
   result_PValue=stats.f_oneway(Hormonizom[Hormonizom.Cluster==1][Col]
                                    ,Hormonizom[Hormonizom.Cluster==2][Col]
                                    ,Hormonizom[Hormonizom.Cluster==3][Col]
                                    ,Hormonizom[Hormonizom.Cluster==4][Col]
                                    ,Hormonizom[Hormonizom.Cluster==5][Col]
                                    ,Hormonizom[Hormonizom.Cluster==6][Col]
                                    ,Hormonizom[Hormonizom.Cluster==7][Col]
                                    ,Hormonizom[Hormonizom.Cluster==8][Col]
                                    ,Hormonizom[Hormonizom.Cluster==9][Col]
                                    ,Hormonizom[Hormonizom.Cluster==10][Col]
                                    ,Hormonizom[Hormonizom.Cluster==11][Col]
                                    ,Hormonizom[Hormonizom.Cluster==12][Col])[1]
   if result_PValue<=0.05:
       HormonizomAnova.at[j,'HormonizomFeature']=Col
       HormonizomAnova.at[j,'FStatistic']=np.round(result_Statistic,4)
       HormonizomAnova.at[j,'Pvalue']=np.round(result_PValue,4)

HormonizomAnova.to_csv('/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/Cluster Overlap By Class/all cluster/Anova_Hormonizom_AllCluster.csv',sep="\t",index=False)
   

####### Mean of SwissADME for even cluster  merge 7 cluster #######
L4=pd.read_csv('/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/Out_L4.csv',delimiter="\t")
L4.loc[L4.PredictClass== 2,'PredictClass'] = 1
L4.loc[L4.PredictClass== 3,'PredictClass'] = 2
L4.loc[L4.PredictClass== 4,'PredictClass'] = 3
L4.loc[L4.PredictClass== 5,'PredictClass'] = 4
L4.loc[L4.PredictClass== 6,'PredictClass'] = 5
L4.loc[L4.PredictClass== 7,'PredictClass'] = 6
L4.loc[L4.PredictClass== 10,'PredictClass'] = 7
L4.loc[L4.PredictClass== 11,'PredictClass'] = 8
L4.loc[L4.PredictClass== 12,'PredictClass'] = 9
L4.loc[L4.PredictClass== 13,'PredictClass'] = 10
L4.loc[L4.PredictClass== 14,'PredictClass'] = 11
L4.loc[L4.PredictClass== 15,'PredictClass'] = 12
L4_WithoutNoClass=L4[L4.Class!='NoClass'].reset_index(drop=True)
L4_WithoutNoClass["Class"] = pd.to_numeric(L4_WithoutNoClass["Class"])
L4_WithoutNoClass_Merge=L4_WithoutNoClass


L4_WithoutNoClass_Merge.loc[L4_WithoutNoClass_Merge.PredictClass== 1,'PredictClass'] = 'a'
L4_WithoutNoClass_Merge.loc[L4_WithoutNoClass_Merge.PredictClass== 2,'PredictClass'] = 'b'
L4_WithoutNoClass_Merge.loc[L4_WithoutNoClass_Merge.PredictClass== 4,'PredictClass'] = 'c'
L4_WithoutNoClass_Merge.loc[L4_WithoutNoClass_Merge.PredictClass== 6,'PredictClass'] = 'a'
L4_WithoutNoClass_Merge.loc[L4_WithoutNoClass_Merge.PredictClass== 11,'PredictClass'] = 'd'
L4_WithoutNoClass_Merge.loc[L4_WithoutNoClass_Merge.PredictClass== 12,'PredictClass'] = 'd'
L4_WithoutNoClass_Merge.loc[L4_WithoutNoClass_Merge.PredictClass== 8,'PredictClass'] = 'e'
L4_WithoutNoClass_Merge.loc[L4_WithoutNoClass_Merge.PredictClass== 7,'PredictClass'] = 'e'
L4_WithoutNoClass_Merge.loc[L4_WithoutNoClass_Merge.PredictClass== 10,'PredictClass'] = 'f'
L4_WithoutNoClass_Merge.loc[L4_WithoutNoClass_Merge.PredictClass== 5,'PredictClass'] = 'f'
L4_WithoutNoClass_Merge.loc[L4_WithoutNoClass_Merge.PredictClass== 9,'PredictClass'] = 'g'
L4_WithoutNoClass_Merge.loc[L4_WithoutNoClass_Merge.PredictClass== 3,'PredictClass'] = 'g'

All_Drug_Swissadme_L4_a=All_Drug_Swissadme[All_Drug_Swissadme.Molecule.isin(L4_WithoutNoClass_Merge[L4_WithoutNoClass_Merge.PredictClass=='a']['CID'])]
All_Drug_Swissadme_L4_b=All_Drug_Swissadme[All_Drug_Swissadme.Molecule.isin(L4_WithoutNoClass_Merge[L4_WithoutNoClass_Merge.PredictClass=='b']['CID'])]
All_Drug_Swissadme_L4_c=All_Drug_Swissadme[All_Drug_Swissadme.Molecule.isin(L4_WithoutNoClass_Merge[L4_WithoutNoClass_Merge.PredictClass=='c']['CID'])]
All_Drug_Swissadme_L4_d=All_Drug_Swissadme[All_Drug_Swissadme.Molecule.isin(L4_WithoutNoClass_Merge[L4_WithoutNoClass_Merge.PredictClass=='d']['CID'])]
All_Drug_Swissadme_L4_e=All_Drug_Swissadme[All_Drug_Swissadme.Molecule.isin(L4_WithoutNoClass_Merge[L4_WithoutNoClass_Merge.PredictClass=='e']['CID'])]
All_Drug_Swissadme_L4_f=All_Drug_Swissadme[All_Drug_Swissadme.Molecule.isin(L4_WithoutNoClass_Merge[L4_WithoutNoClass_Merge.PredictClass=='f']['CID'])]
All_Drug_Swissadme_L4_g=All_Drug_Swissadme[All_Drug_Swissadme.Molecule.isin(L4_WithoutNoClass_Merge[L4_WithoutNoClass_Merge.PredictClass=='g']['CID'])]

All_Drug_Swissadme_L4_a.insert(0,'Cluster','a')
All_Drug_Swissadme_L4_b.insert(0,'Cluster','b')
All_Drug_Swissadme_L4_c.insert(0,'Cluster','c')
All_Drug_Swissadme_L4_d.insert(0,'Cluster','d')
All_Drug_Swissadme_L4_e.insert(0,'Cluster','e')
All_Drug_Swissadme_L4_f.insert(0,'Cluster','f')
All_Drug_Swissadme_L4_g.insert(0,'Cluster','g')

All_Drug_Swissadme_ByClass=All_Drug_Swissadme_L4_a.append(All_Drug_Swissadme_L4_b)
All_Drug_Swissadme_ByClass=All_Drug_Swissadme_ByClass.append(All_Drug_Swissadme_L4_c)
All_Drug_Swissadme_ByClass=All_Drug_Swissadme_ByClass.append(All_Drug_Swissadme_L4_d)
All_Drug_Swissadme_ByClass=All_Drug_Swissadme_ByClass.append(All_Drug_Swissadme_L4_e)
All_Drug_Swissadme_ByClass=All_Drug_Swissadme_ByClass.append(All_Drug_Swissadme_L4_f)
All_Drug_Swissadme_ByClass=All_Drug_Swissadme_ByClass.append(All_Drug_Swissadme_L4_g)

### BoxPlot
sns.boxplot(data=All_Drug_Swissadme_ByClass,x='Cluster',y='MW')
sns.boxplot(data=All_Drug_Swissadme_ByClass,x='Cluster',y='#Heavy atoms')
sns.boxplot(data=All_Drug_Swissadme_ByClass,x='Cluster',y='#Aromatic heavy atoms')
sns.boxplot(data=All_Drug_Swissadme_ByClass,x='Cluster',y='Fraction Csp3')
sns.boxplot(data=All_Drug_Swissadme_ByClass,x='Cluster',y='#Rotatable bonds')
sns.boxplot(data=All_Drug_Swissadme_ByClass,x='Cluster',y='#H-bond acceptors')
sns.boxplot(data=All_Drug_Swissadme_ByClass,x='Cluster',y='#H-bond donors')
sns.boxplot(data=All_Drug_Swissadme_ByClass,x='Cluster',y='MR')
sns.boxplot(data=All_Drug_Swissadme_ByClass,x='Cluster',y='TPSA')
sns.boxplot(data=All_Drug_Swissadme_ByClass,x='Cluster',y='iLOGP')
sns.boxplot(data=All_Drug_Swissadme_ByClass,x='Cluster',y='XLOGP3')
sns.boxplot(data=All_Drug_Swissadme_ByClass,x='Cluster',y='WLOGP')
sns.boxplot(data=All_Drug_Swissadme_ByClass,x='Cluster',y='MLOGP')
sns.boxplot(data=All_Drug_Swissadme_ByClass,x='Cluster',y='Silicos-IT Log P')
sns.boxplot(data=All_Drug_Swissadme_ByClass,x='Cluster',y='Synthetic Accessibility')


##### SwissADME distplot
sns.distplot(All_Drug_Swissadme_L4_a['Fraction Csp3'], kde=True, hist=False,kde_kws={ "label": "Super cluster a"})
sns.distplot(All_Drug_Swissadme_L4_b['Fraction Csp3'], kde=True, hist=False,kde_kws={ "label": "Super cluster b"})
sns.distplot(All_Drug_Swissadme_L4_c['Fraction Csp3'], kde=True, hist=False,kde_kws={"label": "Super cluster c"})
sns.distplot(All_Drug_Swissadme_L4_d['Fraction Csp3'], kde=True, hist=False,kde_kws={ "label": "Super cluster d"})
sns.distplot(All_Drug_Swissadme_L4_e['Fraction Csp3'], kde=True, hist=False,kde_kws={"label": "Super cluster e"})
sns.distplot(All_Drug_Swissadme_L4_g['Fraction Csp3'], kde=True, hist=False,kde_kws={ "label": "Super cluster f"})
sns.distplot(All_Drug_Swissadme_L4_f['Fraction Csp3'], kde=True, hist=False,kde_kws={"label": "Super cluster g"})

sns.distplot(All_Drug_Swissadme_L4_a['iLOGP'], kde=True, hist=False,kde_kws={"label": "Super cluster a"})
sns.distplot(All_Drug_Swissadme_L4_b['iLOGP'], kde=True, hist=False,kde_kws={ "label": "Super cluster b"})
sns.distplot(All_Drug_Swissadme_L4_c['iLOGP'], kde=True, hist=False,kde_kws={"label": "Super cluster c"})
sns.distplot(All_Drug_Swissadme_L4_d['iLOGP'], kde=True, hist=False,kde_kws={ "label": "Super cluster d"})
sns.distplot(All_Drug_Swissadme_L4_e['iLOGP'], kde=True, hist=False,kde_kws={ "label": "Super cluster e"})
sns.distplot(All_Drug_Swissadme_L4_f['iLOGP'], kde=True, hist=False,kde_kws={"label": "Super cluster f"})
sns.distplot(All_Drug_Swissadme_L4_g['iLOGP'], kde=True, hist=False,kde_kws={ "label": "Super cluster g"})

sns.distplot(All_Drug_Swissadme_L4_a['TPSA'], kde=True, hist=False,kde_kws={ "label": "Super cluster a"})
sns.distplot(All_Drug_Swissadme_L4_b['TPSA'], kde=True, hist=False,kde_kws={ "label": "Super cluster b"})
sns.distplot(All_Drug_Swissadme_L4_c['TPSA'], kde=True, hist=False,kde_kws={ "label": "Super cluster c"})
sns.distplot(All_Drug_Swissadme_L4_d['TPSA'], kde=True, hist=False,kde_kws={ "label": "Super cluster d"})
sns.distplot(All_Drug_Swissadme_L4_e['TPSA'], kde=True, hist=False,kde_kws={ "label": "Super cluster e"})
sns.distplot(All_Drug_Swissadme_L4_f['TPSA'], kde=True, hist=False,kde_kws={ "label": "Super cluster f"})
sns.distplot(All_Drug_Swissadme_L4_g['TPSA'], kde=True, hist=False,kde_kws={"label": "Super cluster g"})

sns.distplot(All_Drug_Swissadme_L4_a['#Aromatic heavy atoms'], kde=True, hist=False,kde_kws={"label": "Super cluster a"})
sns.distplot(All_Drug_Swissadme_L4_b['#Aromatic heavy atoms'], kde=True, hist=False,kde_kws={ "label": "Super cluster b"})
sns.distplot(All_Drug_Swissadme_L4_c['#Aromatic heavy atoms'], kde=True, hist=False,kde_kws={ "label": "Super cluster c"})
sns.distplot(All_Drug_Swissadme_L4_d['#Aromatic heavy atoms'], kde=True, hist=False,kde_kws={ "label": "Super cluster d"})
sns.distplot(All_Drug_Swissadme_L4_e['#Aromatic heavy atoms'], kde=True, hist=False,kde_kws={ "label": "Super cluster e"})
sns.distplot(All_Drug_Swissadme_L4_f['#Aromatic heavy atoms'], kde=True, hist=False,kde_kws={ "label": "Super cluster f"})
sns.distplot(All_Drug_Swissadme_L4_g['#Aromatic heavy atoms'], kde=True, hist=False,kde_kws={ "label": "Super cluster g"})

sns.distplot(All_Drug_Swissadme_L4_a['#Heavy atoms'], kde=True, hist=False,kde_kws={"label": "Super cluster a"})
sns.distplot(All_Drug_Swissadme_L4_b['#Heavy atoms'], kde=True, hist=False,kde_kws={ "label": "Super cluster b"})
sns.distplot(All_Drug_Swissadme_L4_c['#Heavy atoms'], kde=True, hist=False,kde_kws={ "label": "Super cluster c"})
sns.distplot(All_Drug_Swissadme_L4_d['#Heavy atoms'], kde=True, hist=False,kde_kws={ "label": "Super cluster d"})
sns.distplot(All_Drug_Swissadme_L4_e['#Heavy atoms'], kde=True, hist=False,kde_kws={ "label": "Super cluster e"})
sns.distplot(All_Drug_Swissadme_L4_f['#Heavy atoms'], kde=True, hist=False,kde_kws={"label": "Super cluster f"})
sns.distplot(All_Drug_Swissadme_L4_g['#Heavy atoms'], kde=True, hist=False,kde_kws={ "label": "Super cluster g"})

sns.distplot(All_Drug_Swissadme_L4_a['Synthetic Accessibility'], kde=True, hist=False,kde_kws={ "label": "Super cluster a"})
sns.distplot(All_Drug_Swissadme_L4_b['Synthetic Accessibility'], kde=True, hist=False,kde_kws={ "label": "Super cluster b"})
sns.distplot(All_Drug_Swissadme_L4_c['Synthetic Accessibility'], kde=True, hist=False,kde_kws={"label": "Super cluster c"})
sns.distplot(All_Drug_Swissadme_L4_d['Synthetic Accessibility'], kde=True, hist=False,kde_kws={"label": "Super cluster d"})
sns.distplot(All_Drug_Swissadme_L4_e['Synthetic Accessibility'], kde=True, hist=False,kde_kws={ "label": "Super cluster e"})
sns.distplot(All_Drug_Swissadme_L4_f['Synthetic Accessibility'], kde=True, hist=False,kde_kws={ "label": "Super cluster f"})
sns.distplot(All_Drug_Swissadme_L4_g['Synthetic Accessibility'], kde=True, hist=False,kde_kws={ "label": "Super cluster g"})

sns.distplot(All_Drug_Swissadme_L4_a['#Rotatable bonds'], kde=True, hist=False,kde_kws={"label": "Super cluster a"})
sns.distplot(All_Drug_Swissadme_L4_b['#Rotatable bonds'], kde=True, hist=False,kde_kws={"label": "Super cluster b"})
sns.distplot(All_Drug_Swissadme_L4_c['#Rotatable bonds'], kde=True, hist=False,kde_kws={"label": "Super cluster c"})
sns.distplot(All_Drug_Swissadme_L4_d['#Rotatable bonds'], kde=True, hist=False,kde_kws={ "label": "Super cluster d"})
sns.distplot(All_Drug_Swissadme_L4_e['#Rotatable bonds'], kde=True, hist=False,kde_kws={"label": "Super cluster e"})
sns.distplot(All_Drug_Swissadme_L4_f['#Rotatable bonds'], kde=True, hist=False,kde_kws={"label": "Super cluster f"})
sns.distplot(All_Drug_Swissadme_L4_g['#Rotatable bonds'], kde=True, hist=False,kde_kws={ "label": "Super cluster g"})

sns.distplot(All_Drug_Swissadme_L4_a['XLOGP3'], kde=True, hist=False,kde_kws={ "label": "Super cluster a"})
sns.distplot(All_Drug_Swissadme_L4_b['XLOGP3'], kde=True, hist=False,kde_kws={ "label": "Super cluster b"})
sns.distplot(All_Drug_Swissadme_L4_c['XLOGP3'], kde=True, hist=False,kde_kws={ "label": "Super cluster c"})
sns.distplot(All_Drug_Swissadme_L4_d['XLOGP3'], kde=True, hist=False,kde_kws={ "label": "Super cluster d"})
sns.distplot(All_Drug_Swissadme_L4_e['XLOGP3'], kde=True, hist=False,kde_kws={ "label": "Super cluster e"})
sns.distplot(All_Drug_Swissadme_L4_f['XLOGP3'], kde=True, hist=False,kde_kws={"label": "Super cluster f"})
sns.distplot(All_Drug_Swissadme_L4_g['XLOGP3'], kde=True, hist=False,kde_kws={ "label": "Super cluster g"})

sns.distplot(All_Drug_Swissadme_L4_a['WLOGP'], kde=True, hist=False,kde_kws={ "label": "Super cluster a"})
sns.distplot(All_Drug_Swissadme_L4_b['WLOGP'], kde=True, hist=False,kde_kws={ "label": "Super cluster b"})
sns.distplot(All_Drug_Swissadme_L4_c['WLOGP'], kde=True, hist=False,kde_kws={ "label": "Super cluster c"})
sns.distplot(All_Drug_Swissadme_L4_d['WLOGP'], kde=True, hist=False,kde_kws={ "label": "Super cluster d"})
sns.distplot(All_Drug_Swissadme_L4_e['WLOGP'], kde=True, hist=False,kde_kws={ "label": "Super cluster e"})
sns.distplot(All_Drug_Swissadme_L4_f['WLOGP'], kde=True, hist=False,kde_kws={"label": "Super cluster f"})
sns.distplot(All_Drug_Swissadme_L4_g['WLOGP'], kde=True, hist=False,kde_kws={"label": "Super cluster g"})

sns.distplot(All_Drug_Swissadme_L4_a['MLOGP'], kde=True, hist=False,kde_kws={ "label": "Super cluster a"})
sns.distplot(All_Drug_Swissadme_L4_b['MLOGP'], kde=True, hist=False,kde_kws={ "label": "Super cluster b"})
sns.distplot(All_Drug_Swissadme_L4_c['MLOGP'], kde=True, hist=False,kde_kws={ "label": "Super cluster c"})
sns.distplot(All_Drug_Swissadme_L4_d['MLOGP'], kde=True, hist=False,kde_kws={ "label": "Super cluster d"})
sns.distplot(All_Drug_Swissadme_L4_e['MLOGP'], kde=True, hist=False,kde_kws={ "label": "Super cluster e"})
sns.distplot(All_Drug_Swissadme_L4_f['MLOGP'], kde=True, hist=False,kde_kws={ "label": "Super cluster f"})
sns.distplot(All_Drug_Swissadme_L4_g['MLOGP'], kde=True, hist=False,kde_kws={"label": "Super cluster g"})

sns.distplot(All_Drug_Swissadme_L4_a['Silicos-IT Log P'], kde=True, hist=False,kde_kws={"label": "Super cluster a"})
sns.distplot(All_Drug_Swissadme_L4_b['Silicos-IT Log P'], kde=True, hist=False,kde_kws={ "label": "Super cluster b"})
sns.distplot(All_Drug_Swissadme_L4_c['Silicos-IT Log P'], kde=True, hist=False,kde_kws={"label": "Super cluster c"})
sns.distplot(All_Drug_Swissadme_L4_d['Silicos-IT Log P'], kde=True, hist=False,kde_kws={ "label": "Super cluster d"})
sns.distplot(All_Drug_Swissadme_L4_e['Silicos-IT Log P'], kde=True, hist=False,kde_kws={ "label": "Super cluster e"})
sns.distplot(All_Drug_Swissadme_L4_f['Silicos-IT Log P'], kde=True, hist=False,kde_kws={ "label": "Super cluster f"})
sns.distplot(All_Drug_Swissadme_L4_g['Silicos-IT Log P'], kde=True, hist=False,kde_kws={"label": "Super cluster g"})

sns.distplot(All_Drug_Swissadme_L4_a['#H-bond acceptors'], kde=True, hist=False,kde_kws={ "label": "Super cluster a"})
sns.distplot(All_Drug_Swissadme_L4_b['#H-bond acceptors'], kde=True, hist=False,kde_kws={ "label": "Super cluster b"})
sns.distplot(All_Drug_Swissadme_L4_c['#H-bond acceptors'], kde=True, hist=False,kde_kws={ "label": "Super cluster c"})
sns.distplot(All_Drug_Swissadme_L4_d['#H-bond acceptors'], kde=True, hist=False,kde_kws={"label": "Super cluster d"})
sns.distplot(All_Drug_Swissadme_L4_e['#H-bond acceptors'], kde=True, hist=False,kde_kws={ "label": "Super cluster e"})
sns.distplot(All_Drug_Swissadme_L4_f['#H-bond acceptors'], kde=True, hist=False,kde_kws={"label": "Super cluster f"})
sns.distplot(All_Drug_Swissadme_L4_g['#H-bond acceptors'], kde=True, hist=False,kde_kws={ "label": "Super cluster g"})

sns.distplot(All_Drug_Swissadme_L4_a['#H-bond donors'], kde=True, hist=False,kde_kws={ "label": "Super cluster a"})
sns.distplot(All_Drug_Swissadme_L4_b['#H-bond donors'], kde=True, hist=False,kde_kws={ "label": "Super cluster b"})
sns.distplot(All_Drug_Swissadme_L4_c['#H-bond donors'], kde=True, hist=False,kde_kws={ "label": "Super cluster c"})
sns.distplot(All_Drug_Swissadme_L4_d['#H-bond donors'], kde=True, hist=False,kde_kws={ "label": "Super cluster d"})
sns.distplot(All_Drug_Swissadme_L4_e['#H-bond donors'], kde=True, hist=False,kde_kws={ "label": "Super cluster e"})
sns.distplot(All_Drug_Swissadme_L4_f['#H-bond donors'], kde=True, hist=False,kde_kws={ "label": "Super cluster f"})
sns.distplot(All_Drug_Swissadme_L4_g['#H-bond donors'], kde=True, hist=False,kde_kws={ "label": "Super cluster g"})

sns.distplot(All_Drug_Swissadme_L4_a['MR'], kde=True, hist=False,kde_kws={"label": "Super cluster a"})
sns.distplot(All_Drug_Swissadme_L4_b['MR'], kde=True, hist=False,kde_kws={"label": "Super cluster b"})
sns.distplot(All_Drug_Swissadme_L4_c['MR'], kde=True, hist=False,kde_kws={"label": "Super cluster c"})
sns.distplot(All_Drug_Swissadme_L4_d['MR'], kde=True, hist=False,kde_kws={ "label": "Super cluster d"})
sns.distplot(All_Drug_Swissadme_L4_e['MR'], kde=True, hist=False,kde_kws={ "label": "Super cluster e"})
sns.distplot(All_Drug_Swissadme_L4_f['MR'], kde=True, hist=False,kde_kws={ "label": "Super cluster f"})
sns.distplot(All_Drug_Swissadme_L4_g['MR'], kde=True, hist=False,kde_kws={"label": "Super cluster g"})


####### Mean of SwissADME for even cluster  merge 5 cluster #######
###################################################################
L4=pd.read_csv('/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/Out_L4.csv',delimiter="\t")
L4.loc[L4.PredictClass== 2,'PredictClass'] = 1
L4.loc[L4.PredictClass== 3,'PredictClass'] = 2
L4.loc[L4.PredictClass== 4,'PredictClass'] = 3
L4.loc[L4.PredictClass== 5,'PredictClass'] = 4
L4.loc[L4.PredictClass== 6,'PredictClass'] = 5
L4.loc[L4.PredictClass== 7,'PredictClass'] = 6
L4.loc[L4.PredictClass== 10,'PredictClass'] = 7
L4.loc[L4.PredictClass== 11,'PredictClass'] = 8
L4.loc[L4.PredictClass== 12,'PredictClass'] = 9
L4.loc[L4.PredictClass== 13,'PredictClass'] = 10
L4.loc[L4.PredictClass== 14,'PredictClass'] = 11
L4.loc[L4.PredictClass== 15,'PredictClass'] = 12
L4_WithoutNoClass=L4[L4.Class!='NoClass'].reset_index(drop=True)
L4_WithoutNoClass["Class"] = pd.to_numeric(L4_WithoutNoClass["Class"])
L4_WithoutNoClass_Merge=L4_WithoutNoClass

L4_WithoutNoClass_Merge.loc[L4_WithoutNoClass_Merge.PredictClass== 1,'PredictClass'] = 'A'
L4_WithoutNoClass_Merge.loc[L4_WithoutNoClass_Merge.PredictClass== 2,'PredictClass'] = 'B'
L4_WithoutNoClass_Merge.loc[L4_WithoutNoClass_Merge.PredictClass== 4,'PredictClass'] = 'C'
L4_WithoutNoClass_Merge.loc[L4_WithoutNoClass_Merge.PredictClass== 6,'PredictClass'] = 'A'
L4_WithoutNoClass_Merge.loc[L4_WithoutNoClass_Merge.PredictClass== 11,'PredictClass'] = 'D'
L4_WithoutNoClass_Merge.loc[L4_WithoutNoClass_Merge.PredictClass== 12,'PredictClass'] = 'D'
L4_WithoutNoClass_Merge.loc[L4_WithoutNoClass_Merge.PredictClass== 8,'PredictClass'] = 'C'
L4_WithoutNoClass_Merge.loc[L4_WithoutNoClass_Merge.PredictClass== 7,'PredictClass'] = 'C'
L4_WithoutNoClass_Merge.loc[L4_WithoutNoClass_Merge.PredictClass== 10,'PredictClass'] = 'E'
L4_WithoutNoClass_Merge.loc[L4_WithoutNoClass_Merge.PredictClass== 5,'PredictClass'] = 'E'
L4_WithoutNoClass_Merge.loc[L4_WithoutNoClass_Merge.PredictClass== 9,'PredictClass'] = 'E'
L4_WithoutNoClass_Merge.loc[L4_WithoutNoClass_Merge.PredictClass== 3,'PredictClass'] = 'E'


All_Drug_Swissadme_L4_A=All_Drug_Swissadme[All_Drug_Swissadme.Molecule.isin(L4_WithoutNoClass_Merge[L4_WithoutNoClass_Merge.PredictClass=='A']['CID'])]
All_Drug_Swissadme_L4_B=All_Drug_Swissadme[All_Drug_Swissadme.Molecule.isin(L4_WithoutNoClass_Merge[L4_WithoutNoClass_Merge.PredictClass=='B']['CID'])]
All_Drug_Swissadme_L4_C=All_Drug_Swissadme[All_Drug_Swissadme.Molecule.isin(L4_WithoutNoClass_Merge[L4_WithoutNoClass_Merge.PredictClass=='C']['CID'])]
All_Drug_Swissadme_L4_D=All_Drug_Swissadme[All_Drug_Swissadme.Molecule.isin(L4_WithoutNoClass_Merge[L4_WithoutNoClass_Merge.PredictClass=='D']['CID'])]
All_Drug_Swissadme_L4_E=All_Drug_Swissadme[All_Drug_Swissadme.Molecule.isin(L4_WithoutNoClass_Merge[L4_WithoutNoClass_Merge.PredictClass=='E']['CID'])]


All_Drug_Swissadme_L4_A.insert(0,'Cluster','A')
All_Drug_Swissadme_L4_B.insert(0,'Cluster','B')
All_Drug_Swissadme_L4_C.insert(0,'Cluster','C')
All_Drug_Swissadme_L4_D.insert(0,'Cluster','D')
All_Drug_Swissadme_L4_E.insert(0,'Cluster','E')

All_Drug_Swissadme_ByClass=All_Drug_Swissadme_L4_A.append(All_Drug_Swissadme_L4_B)
All_Drug_Swissadme_ByClass=All_Drug_Swissadme_ByClass.append(All_Drug_Swissadme_L4_C)
All_Drug_Swissadme_ByClass=All_Drug_Swissadme_ByClass.append(All_Drug_Swissadme_L4_D)
All_Drug_Swissadme_ByClass=All_Drug_Swissadme_ByClass.append(All_Drug_Swissadme_L4_E)


###Anova Test For 5 Cluster
SwissADMEAnova=pd.DataFrame(columns=['Feature','FStatistic','Pvalue'])
for i in range(13):
   j=len(SwissADMEAnova)
   result_Statistic=stats.f_oneway(All_Drug_Swissadme_ByClass[All_Drug_Swissadme_ByClass.Cluster=='A'][PhysicoArr[i]]
                                                   ,All_Drug_Swissadme_ByClass[All_Drug_Swissadme_ByClass.Cluster=='B'][PhysicoArr[i]]
                                                   ,All_Drug_Swissadme_ByClass[All_Drug_Swissadme_ByClass.Cluster=='C'][PhysicoArr[i]]
                                                   ,All_Drug_Swissadme_ByClass[All_Drug_Swissadme_ByClass.Cluster=='D'][PhysicoArr[i]]
                                                   ,All_Drug_Swissadme_ByClass[All_Drug_Swissadme_ByClass.Cluster=='E'][PhysicoArr[i]])[0]
    
   result_Statistic=stats.f_oneway(All_Drug_Swissadme_ByClass[All_Drug_Swissadme_ByClass.Cluster=='A'][PhysicoArr[i]]
                                                   ,All_Drug_Swissadme_ByClass[All_Drug_Swissadme_ByClass.Cluster=='B'][PhysicoArr[i]]
                                                   ,All_Drug_Swissadme_ByClass[All_Drug_Swissadme_ByClass.Cluster=='C'][PhysicoArr[i]]
                                                   ,All_Drug_Swissadme_ByClass[All_Drug_Swissadme_ByClass.Cluster=='D'][PhysicoArr[i]]
                                                   ,All_Drug_Swissadme_ByClass[All_Drug_Swissadme_ByClass.Cluster=='E'][PhysicoArr[i]])[1]
   
   SwissADMEAnova.at[j,'Feature']=PhysicoArr[i]
   SwissADMEAnova.at[j,'FStatistic']=np.round(result_Statistic,4)
   SwissADMEAnova.at[j,'Pvalue']=np.round(result_PValue,4)

SwissADMEAnova.to_csv('/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/Cluster Overlap By Class/all cluster/Anova_Swissadme_AllCluster.csv',sep="\t",index=False)

####### BoxPlot For 5 Cluster

sns.boxplot(data=All_Drug_Swissadme_ByClass,x='Cluster',y='MW')
sns.boxplot(data=All_Drug_Swissadme_ByClass,x='Cluster',y='#Heavy atoms')
sns.boxplot(data=All_Drug_Swissadme_ByClass,x='Cluster',y='#Aromatic heavy atoms')
sns.boxplot(data=All_Drug_Swissadme_ByClass,x='Cluster',y='Fraction Csp3')
sns.boxplot(data=All_Drug_Swissadme_ByClass,x='Cluster',y='#Rotatable bonds')
sns.boxplot(data=All_Drug_Swissadme_ByClass,x='Cluster',y='#H-bond acceptors')
sns.boxplot(data=All_Drug_Swissadme_ByClass,x='Cluster',y='#H-bond donors')
sns.boxplot(data=All_Drug_Swissadme_ByClass,x='Cluster',y='MR')
sns.boxplot(data=All_Drug_Swissadme_ByClass,x='Cluster',y='TPSA')
sns.boxplot(data=All_Drug_Swissadme_ByClass,x='Cluster',y='iLOGP')
sns.boxplot(data=All_Drug_Swissadme_ByClass,x='Cluster',y='XLOGP3')
sns.boxplot(data=All_Drug_Swissadme_ByClass,x='Cluster',y='WLOGP')
sns.boxplot(data=All_Drug_Swissadme_ByClass,x='Cluster',y='MLOGP')
sns.boxplot(data=All_Drug_Swissadme_ByClass,x='Cluster',y='Silicos-IT Log P')
sns.boxplot(data=All_Drug_Swissadme_ByClass,x='Cluster',y='Synthetic Accessibility')



All_Drug_Swissadme_A=All_Drug_Swissadme[All_Drug_Swissadme.Molecule.isin(L4_WithoutNoClass_Merge[L4_WithoutNoClass_Merge.PredictClass=='A']['CID'])]
All_Drug_Swissadme_B=All_Drug_Swissadme[All_Drug_Swissadme.Molecule.isin(L4_WithoutNoClass_Merge[L4_WithoutNoClass_Merge.PredictClass=='B']['CID'])]
All_Drug_Swissadme_C=All_Drug_Swissadme[All_Drug_Swissadme.Molecule.isin(L4_WithoutNoClass_Merge[L4_WithoutNoClass_Merge.PredictClass=='C']['CID'])]
All_Drug_Swissadme_D=All_Drug_Swissadme[All_Drug_Swissadme.Molecule.isin(L4_WithoutNoClass_Merge[L4_WithoutNoClass_Merge.PredictClass=='D']['CID'])]
All_Drug_Swissadme_E=All_Drug_Swissadme[All_Drug_Swissadme.Molecule.isin(L4_WithoutNoClass_Merge[L4_WithoutNoClass_Merge.PredictClass=='E']['CID'])]


sns.distplot(All_Drug_Swissadme_A['Fraction Csp3'], kde=True, hist=False,kde_kws={"label": "Super cluster A"})
sns.distplot(All_Drug_Swissadme_B['Fraction Csp3'], kde=True, hist=False,kde_kws={"label": "Super cluster B"})
sns.distplot(All_Drug_Swissadme_C['Fraction Csp3'], kde=True, hist=False,kde_kws={ "label": "Super cluster C"})
sns.distplot(All_Drug_Swissadme_D['Fraction Csp3'], kde=True, hist=False,kde_kws={"label": "Super cluster D"})
sns.distplot(All_Drug_Swissadme_E['Fraction Csp3'], kde=True, hist=False,kde_kws={"label": "Super cluster E"})

sns.distplot(All_Drug_Swissadme_A['iLOGP'], kde=True, hist=False,kde_kws={ "label": "Super cluster A"})
sns.distplot(All_Drug_Swissadme_B['iLOGP'], kde=True, hist=False,kde_kws={ "label": "Super cluster B"})
sns.distplot(All_Drug_Swissadme_C['iLOGP'], kde=True, hist=False,kde_kws={ "label": "Super cluster C"})
sns.distplot(All_Drug_Swissadme_D['iLOGP'], kde=True, hist=False,kde_kws={ "label": "Super cluster D"})
sns.distplot(All_Drug_Swissadme_E['iLOGP'], kde=True, hist=False,kde_kws={ "label": "Super cluster E"})

sns.distplot(All_Drug_Swissadme_A['TPSA'], kde=True, hist=False,kde_kws={ "label": "Super cluster A"})
sns.distplot(All_Drug_Swissadme_B['TPSA'], kde=True, hist=False,kde_kws={ "label": "Super cluster B"})
sns.distplot(All_Drug_Swissadme_C['TPSA'], kde=True, hist=False,kde_kws={ "label": "Super cluster C"})
sns.distplot(All_Drug_Swissadme_D['TPSA'], kde=True, hist=False,kde_kws={"label": "Super cluster D"})
sns.distplot(All_Drug_Swissadme_E['TPSA'], kde=True, hist=False,kde_kws={"label": "Super cluster E"})

sns.distplot(All_Drug_Swissadme_A['#Aromatic heavy atoms'], kde=True, hist=False,kde_kws={ "label": "Super cluster A"})
sns.distplot(All_Drug_Swissadme_B['#Aromatic heavy atoms'], kde=True, hist=False,kde_kws={ "label": "Super cluster B"})
sns.distplot(All_Drug_Swissadme_C['#Aromatic heavy atoms'], kde=True, hist=False,kde_kws={ "label": "Super cluster C"})
sns.distplot(All_Drug_Swissadme_D['#Aromatic heavy atoms'], kde=True, hist=False,kde_kws={"label": "Super cluster D"})
sns.distplot(All_Drug_Swissadme_E['#Aromatic heavy atoms'], kde=True, hist=False,kde_kws={"label": "Super cluster E"})

sns.distplot(All_Drug_Swissadme_A['#Heavy atoms'], kde=True, hist=False,kde_kws={"label": "Super cluster A"})
sns.distplot(All_Drug_Swissadme_B['#Heavy atoms'], kde=True, hist=False,kde_kws={ "label": "Super cluster B"})
sns.distplot(All_Drug_Swissadme_C['#Heavy atoms'], kde=True, hist=False,kde_kws={ "label": "Super cluster C"})
sns.distplot(All_Drug_Swissadme_D['#Heavy atoms'], kde=True, hist=False,kde_kws={ "label": "Super cluster D"})
sns.distplot(All_Drug_Swissadme_E['#Heavy atoms'], kde=True, hist=False,kde_kws={ "label": "Super cluster E"})

sns.distplot(All_Drug_Swissadme_A['Synthetic Accessibility'], kde=True, hist=False,kde_kws={ "label": "Super cluster A"})
sns.distplot(All_Drug_Swissadme_B['Synthetic Accessibility'], kde=True, hist=False,kde_kws={ "label": "Super cluster B"})
sns.distplot(All_Drug_Swissadme_C['Synthetic Accessibility'], kde=True, hist=False,kde_kws={"label": "Super cluster C"})
sns.distplot(All_Drug_Swissadme_D['Synthetic Accessibility'], kde=True, hist=False,kde_kws={ "label": "Super cluster D"})
sns.distplot(All_Drug_Swissadme_E['Synthetic Accessibility'], kde=True, hist=False,kde_kws={ "label": "Super cluster E"})

sns.distplot(All_Drug_Swissadme_A['#Rotatable bonds'], kde=True, hist=False,kde_kws={"label": "Super cluster A"})
sns.distplot(All_Drug_Swissadme_B['#Rotatable bonds'], kde=True, hist=False,kde_kws={"label": "Super cluster B"})
sns.distplot(All_Drug_Swissadme_C['#Rotatable bonds'], kde=True, hist=False,kde_kws={ "label": "Super cluster C"})
sns.distplot(All_Drug_Swissadme_D['#Rotatable bonds'], kde=True, hist=False,kde_kws={ "label": "Super cluster D"})
sns.distplot(All_Drug_Swissadme_E['#Rotatable bonds'], kde=True, hist=False,kde_kws={ "label": "Super cluster E"})

sns.distplot(All_Drug_Swissadme_A['XLOGP3'], kde=True, hist=False,kde_kws={ "label": "Super cluster A"})
sns.distplot(All_Drug_Swissadme_B['XLOGP3'], kde=True, hist=False,kde_kws={"label": "Super cluster B"})
sns.distplot(All_Drug_Swissadme_C['XLOGP3'], kde=True, hist=False,kde_kws={ "label": "Super cluster C"})
sns.distplot(All_Drug_Swissadme_D['XLOGP3'], kde=True, hist=False,kde_kws={ "label": "Super cluster D"})
sns.distplot(All_Drug_Swissadme_E['XLOGP3'], kde=True, hist=False,kde_kws={ "label": "Super cluster E"})

sns.distplot(All_Drug_Swissadme_A['WLOGP'], kde=True, hist=False,kde_kws={ "label": "Super cluster A"})
sns.distplot(All_Drug_Swissadme_B['WLOGP'], kde=True, hist=False,kde_kws={ "label": "Super cluster B"})
sns.distplot(All_Drug_Swissadme_C['WLOGP'], kde=True, hist=False,kde_kws={ "label": "Super cluster C"})
sns.distplot(All_Drug_Swissadme_D['WLOGP'], kde=True, hist=False,kde_kws={ "label": "Super cluster D"})
sns.distplot(All_Drug_Swissadme_E['WLOGP'], kde=True, hist=False,kde_kws={ "label": "Super cluster E"})

sns.distplot(All_Drug_Swissadme_A['MLOGP'], kde=True, hist=False,kde_kws={ "label": "Super cluster A"})
sns.distplot(All_Drug_Swissadme_B['MLOGP'], kde=True, hist=False,kde_kws={"label": "Super cluster B"})
sns.distplot(All_Drug_Swissadme_C['MLOGP'], kde=True, hist=False,kde_kws={"label": "Super cluster C"})
sns.distplot(All_Drug_Swissadme_D['MLOGP'], kde=True, hist=False,kde_kws={"label": "Super cluster D"})
sns.distplot(All_Drug_Swissadme_E['MLOGP'], kde=True, hist=False,kde_kws={"label": "Super cluster E"})

sns.distplot(All_Drug_Swissadme_A['Silicos-IT Log P'], kde=True, hist=False,kde_kws={ "label": "Super cluster A"})
sns.distplot(All_Drug_Swissadme_B['Silicos-IT Log P'], kde=True, hist=False,kde_kws={ "label": "Super cluster B"})
sns.distplot(All_Drug_Swissadme_C['Silicos-IT Log P'], kde=True, hist=False,kde_kws={"label": "Super cluster C"})
sns.distplot(All_Drug_Swissadme_D['Silicos-IT Log P'], kde=True, hist=False,kde_kws={"label": "Super cluster D"})
sns.distplot(All_Drug_Swissadme_E['Silicos-IT Log P'], kde=True, hist=False,kde_kws={"label": "Super cluster E"})

sns.distplot(All_Drug_Swissadme_A['#H-bond acceptors'], kde=True, hist=False,kde_kws={ "label": "Super cluster A"})
sns.distplot(All_Drug_Swissadme_B['#H-bond acceptors'], kde=True, hist=False,kde_kws={ "label": "Super cluster B"})
sns.distplot(All_Drug_Swissadme_C['#H-bond acceptors'], kde=True, hist=False,kde_kws={ "label": "Super cluster C"})
sns.distplot(All_Drug_Swissadme_D['#H-bond acceptors'], kde=True, hist=False,kde_kws={ "label": "Super cluster D"})
sns.distplot(All_Drug_Swissadme_E['#H-bond acceptors'], kde=True, hist=False,kde_kws={ "label": "Super cluster E"})

sns.distplot(All_Drug_Swissadme_A['#H-bond donors'], kde=True, hist=False,kde_kws={"label": "Super cluster A"})
sns.distplot(All_Drug_Swissadme_B['#H-bond donors'], kde=True, hist=False,kde_kws={ "label": "Super cluster B"})
sns.distplot(All_Drug_Swissadme_C['#H-bond donors'], kde=True, hist=False,kde_kws={ "label": "Super cluster C"})
sns.distplot(All_Drug_Swissadme_D['#H-bond donors'], kde=True, hist=False,kde_kws={"label": "Super cluster D"})
sns.distplot(All_Drug_Swissadme_E['#H-bond donors'], kde=True, hist=False,kde_kws={"label": "Super cluster E"})

sns.distplot(All_Drug_Swissadme_A['MR'], kde=True, hist=False,kde_kws={ "label": "Super cluster A"})
sns.distplot(All_Drug_Swissadme_B['MR'], kde=True, hist=False,kde_kws={ "label": "Super cluster B"})
sns.distplot(All_Drug_Swissadme_C['MR'], kde=True, hist=False,kde_kws={ "label": "Super cluster C"})
sns.distplot(All_Drug_Swissadme_D['MR'], kde=True, hist=False,kde_kws={"label": "Super cluster D"})
sns.distplot(All_Drug_Swissadme_E['MR'], kde=True, hist=False,kde_kws={"label": "Super cluster E"})


######### Evaluation By Pathway ###############
###############################################
#Data PreProcess #
AllDrugPath=pd.read_csv('/Users/aghil/Desktop/uni/Tese 2/DrugPathway/DrugPathway_from_DrugPath.txt',delimiter="\t")
AllDrugPathCid=pd.read_csv('/Users/aghil/Desktop/uni/Tese 2/DrugPathway/DrugNameCid.txt',delimiter="\t")
AllDrugPathCid=AllDrugPathCid.drop_duplicates().dropna().reset_index(drop=True)

AllDrugPath.insert(0,'CID',0)
for row in AllDrugPath.index:
    CID=AllDrugPathCid[AllDrugPathCid.Name==AllDrugPath.at[row,'DrugName']].drop_duplicates().reset_index(drop=True)
    if len(CID)>0:
        AllDrugPath.at[row,'CID']=CID.at[0,'CID']
AllDrugPath.drop(AllDrugPath[AllDrugPath.CID==0].index,inplace=True)
AllDrugPath.to_csv('/Users/aghil/Desktop/uni/Tese 2/DrugPathway/DrugPathway_from_DrugPath_After_PreProcess.txt',index=False,sep="\t")

AllDrugPath=pd.read_csv('/Users/aghil/Desktop/uni/Tese 2/DrugPathway/DrugPathway_from_DrugPath_After_PreProcess.txt',delimiter="\t")


#Evaluation#
AllDrugPathByCluster=pd.DataFrame(columns=AllDrugPath.columns)
AllDrugPathByCluster.insert(0,'Cluster',np.NaN)
for DrugPathCid in AllDrugPath['CID'].drop_duplicates().reset_index(drop=True):
    AllCluster=pd.DataFrame(Data[Data.CID==DrugPathCid].PredictClass).drop_duplicates().reset_index(drop=True)
    for Cluster in AllCluster.PredictClass:
        AllDrugPathByCluster=AllDrugPathByCluster.append(AllDrugPath[AllDrugPath.CID==DrugPathCid].reset_index(drop=True),ignore_index=True)
        AllDrugPathByCluster=AllDrugPathByCluster.fillna(Cluster)

AllDrugPathByCluster.to_csv('/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/AllDrugPathInEvenCluster.csv',sep="\t",index=False)
AllDrugPathByCluster=pd.read_csv('/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/AllDrugPathInEvenCluster.csv',delimiter="\t")


#### even up or down gene for drug locate in one row #####
def prepend(list, str): 
      
    # Using format() 
    str += '{0}'
    list = [str.format(i) for i in list] 
    return(list) 

count=0
AllDrugPath_Gene=pd.DataFrame(columns=['CID','Cluster','DrugName','Gene','PathwayCode'])
for row in AllDrugPathByCluster.index:
    count=count+1
    print(str(count))
    GeneList=re.split('_', AllDrugPathByCluster.at[row,'GeneID'])
    GeneList = prepend(GeneList, AllDrugPathByCluster.at[row,'UpOrDown'])
    # GeneList=int(GeneList)
    for Gene in GeneList:
        i=len(AllDrugPath_Gene)
        AllDrugPath_Gene.at[i,'CID']=AllDrugPathByCluster.at[row,'CID']
        AllDrugPath_Gene.at[i,'Cluster']=AllDrugPathByCluster.at[row,'Cluster']
        AllDrugPath_Gene.at[i,'DrugName']=AllDrugPathByCluster.at[row,'DrugName']
        AllDrugPath_Gene.at[i,'PathwayCode']=AllDrugPathByCluster.at[row,'PathwayCode']
        AllDrugPath_Gene.at[i,'Gene']=Gene


AllDrugPath_Gene.to_csv('H:/Uni Tese/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/AllDrugPathInEvenCluster.csv',sep="\t",index=False)
AllDrugPath_Gene=pd.read_csv('H:/Uni Tese/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/AllDrugPathInEvenCluster.csv',delimiter="\t")

### distribution Pathway in even cluster
#By 5 Cluster
# AllDrugPath_Gene.loc[AllDrugPath_Gene.Cluster== 1,'Cluster'] = 'A'
# AllDrugPath_Gene.loc[AllDrugPath_Gene.Cluster== 2,'Cluster'] = 'B'
# AllDrugPath_Gene.loc[AllDrugPath_Gene.Cluster== 4,'Cluster'] = 'C'
# AllDrugPath_Gene.loc[AllDrugPath_Gene.Cluster== 6,'Cluster'] = 'A'
# AllDrugPath_Gene.loc[AllDrugPath_Gene.Cluster== 11,'Cluster'] = 'D'
# AllDrugPath_Gene.loc[AllDrugPath_Gene.Cluster== 12,'Cluster'] = 'D'
# AllDrugPath_Gene.loc[AllDrugPath_Gene.Cluster== 8,'Cluster'] = 'C'
# AllDrugPath_Gene.loc[AllDrugPath_Gene.Cluster== 7,'Cluster'] = 'C'
# AllDrugPath_Gene.loc[AllDrugPath_Gene.Cluster== 10,'Cluster'] = 'E'
# AllDrugPath_Gene.loc[AllDrugPath_Gene.Cluster== 5,'Cluster'] = 'E'
# AllDrugPath_Gene.loc[AllDrugPath_Gene.Cluster== 9,'Cluster'] = 'E'
# AllDrugPath_Gene.loc[AllDrugPath_Gene.Cluster== 3,'Cluster'] = 'E'

AllDrugPath_Gene.groupby('Cluster').count()

#By Pathway
np.set_printoptions(suppress=True)  
AllDrugPath_Gene['Cluster']=AllDrugPath_Gene['Cluster'].astype("int")
pivot=AllDrugPath_Gene.pivot_table(index="Cluster",columns='PathwayCode',values='CID',aggfunc="count")
pivot=pivot.fillna(0)
for i in range(0,12):
    pivot.iloc[i:i+1,:]=pivot.iloc[i:i+1,:].div(len(AllDrugPath_Gene[AllDrugPath_Gene.Cluster==i+1].drop_duplicates()))

# pivot.loc['A':'A',:]=pivot.loc['A':'A',:].div(len(AllDrugPath_Gene[AllDrugPath_Gene.Cluster=='A'].drop_duplicates()))
# pivot.loc['B':'B',:]=pivot.loc['B':'B',:].div(len(AllDrugPath_Gene[AllDrugPath_Gene.Cluster=='B'].drop_duplicates()))
# pivot.loc['C':'C',:]=pivot.loc['C':'C',:].div(len(AllDrugPath_Gene[AllDrugPath_Gene.Cluster=='C'].drop_duplicates()))
# pivot.loc['D':'D',:]=pivot.loc['D':'D',:].div(len(AllDrugPath_Gene[AllDrugPath_Gene.Cluster=='D'].drop_duplicates()))
# pivot.loc['E':'E',:]=pivot.loc['E':'E',:].div(len(AllDrugPath_Gene[AllDrugPath_Gene.Cluster=='E'].drop_duplicates()))
plt.figure(figsize=(15,12),dpi=80)
ax=sns.heatmap(pivot.sort_values(by=list(pivot.index),axis=1),yticklabels=pivot.index.sort_values(),cmap="coolwarm",annot=False,fmt='g',vmin=0,vmax=0.1)
ax.set_ylim(12, 0.1)

#By Gene
np.set_printoptions(suppress=True)  
# AllDrugPath_Gene['Cluster']=AllDrugPath_Gene['Cluster'].astype("int")
AllDrugPath_Gene['Gene']=np.abs(AllDrugPath_Gene['Gene'].astype("int"))
AllDrugPath_Gene.loc[AllDrugPath_Gene.Cluster== 1,'Cluster'] = 'A'
AllDrugPath_Gene.loc[AllDrugPath_Gene.Cluster== 2,'Cluster'] = 'B'
AllDrugPath_Gene.loc[AllDrugPath_Gene.Cluster== 4,'Cluster'] = 'C'
AllDrugPath_Gene.loc[AllDrugPath_Gene.Cluster== 6,'Cluster'] = 'A'
AllDrugPath_Gene.loc[AllDrugPath_Gene.Cluster== 11,'Cluster'] = 'D'
AllDrugPath_Gene.loc[AllDrugPath_Gene.Cluster== 12,'Cluster'] = 'D'
AllDrugPath_Gene.loc[AllDrugPath_Gene.Cluster== 8,'Cluster'] = 'C'
AllDrugPath_Gene.loc[AllDrugPath_Gene.Cluster== 7,'Cluster'] = 'C'
AllDrugPath_Gene.loc[AllDrugPath_Gene.Cluster== 10,'Cluster'] = 'E'
AllDrugPath_Gene.loc[AllDrugPath_Gene.Cluster== 5,'Cluster'] = 'E'
AllDrugPath_Gene.loc[AllDrugPath_Gene.Cluster== 9,'Cluster'] = 'E'
AllDrugPath_Gene.loc[AllDrugPath_Gene.Cluster== 3,'Cluster'] = 'E'

pivot=AllDrugPath_Gene.pivot_table(index="Cluster",columns='Gene',values='CID',aggfunc="count")
pivot=pivot.fillna(0)
for i in range(0,12):
    pivot.iloc[i:i+1,:]=pivot.iloc[i:i+1,:].div(len(AllDrugPath_Gene[AllDrugPath_Gene.Cluster==i+1].drop_duplicates()))

# pivot.loc['A':'A',:]=pivot.loc['A':'A',:].div(len(AllDrugPath_Gene[AllDrugPath_Gene.Cluster=='A'].drop_duplicates()))
# pivot.loc['B':'B',:]=pivot.loc['B':'B',:].div(len(AllDrugPath_Gene[AllDrugPath_Gene.Cluster=='B'].drop_duplicates()))
# pivot.loc['C':'C',:]=pivot.loc['C':'C',:].div(len(AllDrugPath_Gene[AllDrugPath_Gene.Cluster=='C'].drop_duplicates()))
# pivot.loc['D':'D',:]=pivot.loc['D':'D',:].div(len(AllDrugPath_Gene[AllDrugPath_Gene.Cluster=='D'].drop_duplicates()))
# pivot.loc['E':'E',:]=pivot.loc['E':'E',:].div(len(AllDrugPath_Gene[AllDrugPath_Gene.Cluster=='E'].drop_duplicates()))

plt.figure(figsize=(15,12),dpi=80)
ax=sns.heatmap(pivot.sort_values(by=list(pivot.index),axis=1),yticklabels=pivot.index.sort_values(),cmap="coolwarm",annot=False,fmt='g',vmin=0,vmax=15)
ax.set_ylim(12, 0.1)

#############
AllDrugPath_Gene[AllDrugPath_Gene.Cluster==1].Gene.to_csv('/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/AllDrugPathInEvenCluster_1.csv',sep="\t",index=False)
AllDrugPath_Gene[AllDrugPath_Gene.Cluster==2].Gene.to_csv('/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/AllDrugPathInEvenCluster_2.csv',sep="\t",index=False)
AllDrugPath_Gene[AllDrugPath_Gene.Cluster==3].Gene.to_csv('/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/AllDrugPathInEvenCluster_3.csv',sep="\t",index=False)
AllDrugPath_Gene[AllDrugPath_Gene.Cluster==4].Gene.to_csv('/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/AllDrugPathInEvenCluster_4.csv',sep="\t",index=False)
AllDrugPath_Gene[AllDrugPath_Gene.Cluster==5].Gene.to_csv('/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/AllDrugPathInEvenCluster_5.csv',sep="\t",index=False)
AllDrugPath_Gene[AllDrugPath_Gene.Cluster==6].Gene.to_csv('/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/AllDrugPathInEvenCluster_6.csv',sep="\t",index=False)
AllDrugPath_Gene[AllDrugPath_Gene.Cluster==7].Gene.to_csv('/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/AllDrugPathInEvenCluster_7.csv',sep="\t",index=False)
AllDrugPath_Gene[AllDrugPath_Gene.Cluster==8].Gene.to_csv('/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/AllDrugPathInEvenCluster_8.csv',sep="\t",index=False)
AllDrugPath_Gene[AllDrugPath_Gene.Cluster==9].Gene.to_csv('/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/AllDrugPathInEvenCluster_9.csv',sep="\t",index=False)
AllDrugPath_Gene[AllDrugPath_Gene.Cluster==10].Gene.to_csv('/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/AllDrugPathInEvenCluster_10.csv',sep="\t",index=False)
AllDrugPath_Gene[AllDrugPath_Gene.Cluster==11].Gene.to_csv('/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/AllDrugPathInEvenCluster_11.csv',sep="\t",index=False)
AllDrugPath_Gene[AllDrugPath_Gene.Cluster==12].Gene.to_csv('/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/AllDrugPathInEvenCluster_12.csv',sep="\t",index=False)


Candid=AllDrugPath_Gene[AllDrugPath_Gene.Cluster==1]
AllClusterExceptCandid=AllDrugPath_Gene[AllDrugPath_Gene.Cluster!=1]
Diff1=Candid[~Candid.Gene.isin(AllClusterExceptCandid.Gene)]


Candid=AllDrugPath_Gene[AllDrugPath_Gene.Cluster==2]
AllClusterExceptCandid=AllDrugPath_Gene[AllDrugPath_Gene.Cluster!=2]
Diff2=Candid[~Candid.Gene.isin(AllClusterExceptCandid.Gene)]


Candid=AllDrugPath_Gene[AllDrugPath_Gene.Cluster==3]
AllClusterExceptCandid=AllDrugPath_Gene[AllDrugPath_Gene.Cluster!=3]
Diff3=Candid[~Candid.Gene.isin(AllClusterExceptCandid.Gene)]


Candid=AllDrugPath_Gene[AllDrugPath_Gene.Cluster==4]
AllClusterExceptCandid=AllDrugPath_Gene[AllDrugPath_Gene.Cluster!=4]
Diff4=Candid[~Candid.Gene.isin(AllClusterExceptCandid.Gene)]


Candid=AllDrugPath_Gene[AllDrugPath_Gene.Cluster==5]
AllClusterExceptCandid=AllDrugPath_Gene[AllDrugPath_Gene.Cluster!=5]
Diff5=Candid[~Candid.Gene.isin(AllClusterExceptCandid.Gene)]


Candid=AllDrugPath_Gene[AllDrugPath_Gene.Cluster==6]
AllClusterExceptCandid=AllDrugPath_Gene[AllDrugPath_Gene.Cluster!=6]
Diff6=Candid[~Candid.Gene.isin(AllClusterExceptCandid.Gene)]


Candid=AllDrugPath_Gene[AllDrugPath_Gene.Cluster==7]
AllClusterExceptCandid=AllDrugPath_Gene[AllDrugPath_Gene.Cluster!=7]
Diff7=Candid[~Candid.Gene.isin(AllClusterExceptCandid.Gene)]


Candid=AllDrugPath_Gene[AllDrugPath_Gene.Cluster==8]
AllClusterExceptCandid=AllDrugPath_Gene[AllDrugPath_Gene.Cluster!=8]
Diff8=Candid[~Candid.Gene.isin(AllClusterExceptCandid.Gene)]



Candid=AllDrugPath_Gene[AllDrugPath_Gene.Cluster==9]
AllClusterExceptCandid=AllDrugPath_Gene[AllDrugPath_Gene.Cluster!=9]
Diff9=Candid[~Candid.Gene.isin(AllClusterExceptCandid.Gene)]


Candid=AllDrugPath_Gene[AllDrugPath_Gene.Cluster==10]
AllClusterExceptCandid=AllDrugPath_Gene[AllDrugPath_Gene.Cluster!=10]
Diff10=Candid[~Candid.Gene.isin(AllClusterExceptCandid.Gene)]

Candid=AllDrugPath_Gene[AllDrugPath_Gene.Cluster==11]
AllClusterExceptCandid=AllDrugPath_Gene[AllDrugPath_Gene.Cluster!=11]
Diff11=Candid[~Candid.Gene.isin(AllClusterExceptCandid.Gene)]

Candid=AllDrugPath_Gene[AllDrugPath_Gene.Cluster==12]
AllClusterExceptCandid=AllDrugPath_Gene[AllDrugPath_Gene.Cluster!=12]
Diff12=Candid[~Candid.Gene.isin(AllClusterExceptCandid.Gene)]

AllDiff=Diff1.append(Diff2)
AllDiff=AllDiff.append(Diff3)
AllDiff=AllDiff.append(Diff4)
AllDiff=AllDiff.append(Diff5)
AllDiff=AllDiff.append(Diff6)
AllDiff=AllDiff.append(Diff7)
AllDiff=AllDiff.append(Diff8)
AllDiff=AllDiff.append(Diff9)
AllDiff=AllDiff.append(Diff10)
AllDiff=AllDiff.append(Diff11)
AllDiff=AllDiff.append(Diff12)

AllDiff.groupby('Cluster').count()

### Merge to 5 Cluster
# AllDrugPath_Gene['Gene']=np.abs(AllDrugPath_Gene['Gene'].astype("int"))
AllDrugPath_Gene=pd.read_csv('H:/Uni Tese/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/AllDrugPathInEvenCluster.csv',delimiter="\t")
AllDrugPath_Gene.loc[AllDrugPath_Gene.Cluster== 1,'Cluster'] = 'A'
AllDrugPath_Gene.loc[AllDrugPath_Gene.Cluster== 2,'Cluster'] = 'A'
AllDrugPath_Gene.loc[AllDrugPath_Gene.Cluster== 3,'Cluster'] = 'B'
AllDrugPath_Gene.loc[AllDrugPath_Gene.Cluster== 4,'Cluster'] = 'B'
AllDrugPath_Gene.loc[AllDrugPath_Gene.Cluster== 5,'Cluster'] = 'C'
AllDrugPath_Gene.loc[AllDrugPath_Gene.Cluster== 6,'Cluster'] = 'C'
AllDrugPath_Gene.loc[AllDrugPath_Gene.Cluster== 7,'Cluster'] = 'D'
AllDrugPath_Gene.loc[AllDrugPath_Gene.Cluster== 8,'Cluster'] = 'E'
AllDrugPath_Gene.loc[AllDrugPath_Gene.Cluster== 9,'Cluster'] = 'F'
AllDrugPath_Gene.loc[AllDrugPath_Gene.Cluster== 10,'Cluster'] = 'G'
AllDrugPath_Gene.loc[AllDrugPath_Gene.Cluster== 11,'Cluster'] = 'G'
AllDrugPath_Gene.loc[AllDrugPath_Gene.Cluster== 12,'Cluster'] = 'H'

AllDrugPath_Gene.insert(0,'GenePathway',AllDrugPath_Gene['Gene'].astype(str)+AllDrugPath_Gene['PathwayCode'])

pivot=AllDrugPath_Gene.pivot_table(index="Cluster",columns='Gene',values='CID',aggfunc="count")
pivot=pivot.fillna(0)

# pivot.loc['A':'A',:]=pivot.loc['A':'A',:].div(len(AllDrugPath_Gene[AllDrugPath_Gene.Cluster=='A'].drop_duplicates()))
# pivot.loc['B':'B',:]=pivot.loc['B':'B',:].div(len(AllDrugPath_Gene[AllDrugPath_Gene.Cluster=='B'].drop_duplicates()))
# pivot.loc['C':'C',:]=pivot.loc['C':'C',:].div(len(AllDrugPath_Gene[AllDrugPath_Gene.Cluster=='C'].drop_duplicates()))
# pivot.loc['D':'D',:]=pivot.loc['D':'D',:].div(len(AllDrugPath_Gene[AllDrugPath_Gene.Cluster=='D'].drop_duplicates()))

plt.figure(figsize=(15,12),dpi=80)
ax=sns.heatmap(pivot.sort_values(by=list(pivot.index),axis=1),yticklabels=pivot.index.sort_values(),cmap="coolwarm",annot=False,fmt='g',vmin=0,vmax=20)
# ax.set_ylim(12, 0.1)

Candid=AllDrugPath_Gene[AllDrugPath_Gene.Cluster=='A']
AllClusterExceptCandid=AllDrugPath_Gene[AllDrugPath_Gene.Cluster!='A']
Diff1=Candid[~Candid.GenePathway.isin(AllClusterExceptCandid.GenePathway)][['Cluster','GenePathway']].drop_duplicates()

Candid=AllDrugPath_Gene[AllDrugPath_Gene.Cluster=='B']
AllClusterExceptCandid=AllDrugPath_Gene[AllDrugPath_Gene.Cluster!='B']
Diff2=Candid[~Candid.GenePathway.isin(AllClusterExceptCandid.GenePathway)][['Cluster','GenePathway']].drop_duplicates()


Candid=AllDrugPath_Gene[AllDrugPath_Gene.Cluster=='C']
AllClusterExceptCandid=AllDrugPath_Gene[AllDrugPath_Gene.Cluster!='C']
Diff3=Candid[~Candid.GenePathway.isin(AllClusterExceptCandid.GenePathway)][['Cluster','GenePathway']].drop_duplicates()


Candid=AllDrugPath_Gene[AllDrugPath_Gene.Cluster=='D']
AllClusterExceptCandid=AllDrugPath_Gene[AllDrugPath_Gene.Cluster!='D']
Diff4=Candid[~Candid.GenePathway.isin(AllClusterExceptCandid.GenePathway)][['Cluster','GenePathway']].drop_duplicates()


Candid=AllDrugPath_Gene[AllDrugPath_Gene.Cluster=='E']
AllClusterExceptCandid=AllDrugPath_Gene[AllDrugPath_Gene.Cluster!='E']
Diff5=Candid[~Candid.GenePathway.isin(AllClusterExceptCandid.GenePathway)][['Cluster','GenePathway']].drop_duplicates()

Candid=AllDrugPath_Gene[AllDrugPath_Gene.Cluster=='F']
AllClusterExceptCandid=AllDrugPath_Gene[AllDrugPath_Gene.Cluster!='F']
Diff6=Candid[~Candid.GenePathway.isin(AllClusterExceptCandid.GenePathway)][['Cluster','GenePathway']].drop_duplicates()

Candid=AllDrugPath_Gene[AllDrugPath_Gene.Cluster=='G']
AllClusterExceptCandid=AllDrugPath_Gene[AllDrugPath_Gene.Cluster!='G']
Diff7=Candid[~Candid.GenePathway.isin(AllClusterExceptCandid.GenePathway)][['Cluster','GenePathway']].drop_duplicates()

Candid=AllDrugPath_Gene[AllDrugPath_Gene.Cluster=='H']
AllClusterExceptCandid=AllDrugPath_Gene[AllDrugPath_Gene.Cluster!='H']
Diff8=Candid[~Candid.GenePathway.isin(AllClusterExceptCandid.GenePathway)][['Cluster','GenePathway']].drop_duplicates()



AllDiff=Diff1.append(Diff2)
AllDiff=AllDiff.append(Diff3)
AllDiff=AllDiff.append(Diff4)
AllDiff=AllDiff.append(Diff5)
AllDiff=AllDiff.append(Diff6)
AllDiff=AllDiff.append(Diff7)
AllDiff=AllDiff.append(Diff8)



AllDiff.groupby('Cluster').count()
########



AllDiff.to_csv('/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/AllGene_Specefic_In_All_Cluster.csv',sep="\t",index=False)
# L4_WithoutNoClass_Merge.loc[L4_WithoutNoClass_Merge.PredictClass== 1,'PredictClass'] = 'a'
# L4_WithoutNoClass_Merge.loc[L4_WithoutNoClass_Merge.PredictClass== 2,'PredictClass'] = 'b'
# L4_WithoutNoClass_Merge.loc[L4_WithoutNoClass_Merge.PredictClass== 4,'PredictClass'] = 'c'
# L4_WithoutNoClass_Merge.loc[L4_WithoutNoClass_Merge.PredictClass== 6,'PredictClass'] = 'a'
# L4_WithoutNoClass_Merge.loc[L4_WithoutNoClass_Merge.PredictClass== 11,'PredictClass'] = 'd'
# L4_WithoutNoClass_Merge.loc[L4_WithoutNoClass_Merge.PredictClass== 12,'PredictClass'] = 'd'
# L4_WithoutNoClass_Merge.loc[L4_WithoutNoClass_Merge.PredictClass== 8,'PredictClass'] = 'e'
# L4_WithoutNoClass_Merge.loc[L4_WithoutNoClass_Merge.PredictClass== 7,'PredictClass'] = 'e'
# L4_WithoutNoClass_Merge.loc[L4_WithoutNoClass_Merge.PredictClass== 10,'PredictClass'] = 'f'
# L4_WithoutNoClass_Merge.loc[L4_WithoutNoClass_Merge.PredictClass== 5,'PredictClass'] = 'f'
# L4_WithoutNoClass_Merge.loc[L4_WithoutNoClass_Merge.PredictClass== 9,'PredictClass'] = 'g'
# L4_WithoutNoClass_Merge.loc[L4_WithoutNoClass_Merge.PredictClass== 3,'PredictClass'] = 'g'


# L4_WithoutNoClass_Merge.loc[L4_WithoutNoClass_Merge.PredictClass== 1,'PredictClass'] = 'A'
# L4_WithoutNoClass_Merge.loc[L4_WithoutNoClass_Merge.PredictClass== 2,'PredictClass'] = 'B'
# L4_WithoutNoClass_Merge.loc[L4_WithoutNoClass_Merge.PredictClass== 4,'PredictClass'] = 'C'
# L4_WithoutNoClass_Merge.loc[L4_WithoutNoClass_Merge.PredictClass== 6,'PredictClass'] = 'A'
# L4_WithoutNoClass_Merge.loc[L4_WithoutNoClass_Merge.PredictClass== 11,'PredictClass'] = 'D'
# L4_WithoutNoClass_Merge.loc[L4_WithoutNoClass_Merge.PredictClass== 12,'PredictClass'] = 'D'
# L4_WithoutNoClass_Merge.loc[L4_WithoutNoClass_Merge.PredictClass== 8,'PredictClass'] = 'C'
# L4_WithoutNoClass_Merge.loc[L4_WithoutNoClass_Merge.PredictClass== 7,'PredictClass'] = 'C'
# L4_WithoutNoClass_Merge.loc[L4_WithoutNoClass_Merge.PredictClass== 10,'PredictClass'] = 'E'
# L4_WithoutNoClass_Merge.loc[L4_WithoutNoClass_Merge.PredictClass== 5,'PredictClass'] = 'E'
# L4_WithoutNoClass_Merge.loc[L4_WithoutNoClass_Merge.PredictClass== 9,'PredictClass'] = 'E'
# L4_WithoutNoClass_Merge.loc[L4_WithoutNoClass_Merge.PredictClass== 3,'PredictClass'] = 'E'

# L4.loc[L4.PredictClass== 2,'PredictClass'] = 1
# L4.loc[L4.PredictClass== 3,'PredictClass'] = 2
# L4.loc[L4.PredictClass== 4,'PredictClass'] = 3
# L4.loc[L4.PredictClass== 5,'PredictClass'] = 4
# L4.loc[L4.PredictClass== 6,'PredictClass'] = 5
# L4.loc[L4.PredictClass== 7,'PredictClass'] = 6
# L4.loc[L4.PredictClass== 10,'PredictClass'] = 7
# L4.loc[L4.PredictClass== 11,'PredictClass'] = 8
# L4.loc[L4.PredictClass== 12,'PredictClass'] = 9
# L4.loc[L4.PredictClass== 13,'PredictClass'] = 10
# L4.loc[L4.PredictClass== 14,'PredictClass'] = 11
# L4.loc[L4.PredictClass== 15,'PredictClass'] = 12

# Cluster27 --  Super Cluster A
# Cluster1415 -- Super Cluster D
# Cluster51011 -- Super Cluster C
# Cluster136124 -- Super Cluster E
# Cluster3 -- Super Cluster B
