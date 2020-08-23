import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib_venn import venn3, venn3_circles
from sklearn.metrics import jaccard_score
from sklearn.metrics import matthews_corrcoef


Hormonizom=pd.read_csv("/Users/aghil/Desktop/uni/Tese 2/final list/Train_By Common Multimodal Data/only hormonizom/each Cell Line Dose And Time In One Row/All_Hormonizom_Col_Lower_VarQuantile0.75_Remove.csv",delimiter="\t")
Hormonizom=Hormonizom.dropna()
Hormonizom_Candid=Hormonizom[Hormonizom.CID.isin([3652,92727,41684,3117])].reset_index(drop=True)
 
##### Calculate jacccard for Hormonizom  ##############
####################################################
count=0
HormonizomSimilaity=pd.DataFrame(columns=['Code1','CID1','ClassCode1','Code2','CID2','ClassCode2','Similarity'])
# i=0
for i in Hormonizom_Candid.index:
    for j in Hormonizom.index:
        count=count+1
        print(str(count))
        CID1=Hormonizom_Candid.at[i,'CID']
        CID2=Hormonizom.at[j,'CID']
        Code1=Hormonizom_Candid.at[i,'Code']
        Code2=Hormonizom.at[j,'Code']        
        row=len(HormonizomSimilaity)
        HormonizomSimilaity.at[row,'CID1']=CID1
        HormonizomSimilaity.at[row,'CID2']=CID2
        HormonizomSimilaity.at[row,'Code1']=Code1
        HormonizomSimilaity.at[row,'Code2']=Code2        
        HormonizomSimilaity.at[row,'ClassCode1']=Hormonizom_Candid.at[i,'Class']
        HormonizomSimilaity.at[row,'ClassCode2']=Hormonizom.at[j,'Class']
        HormonizomSimilaity.at[row,'Similarity']=round(jaccard_score(np.ravel(Hormonizom_Candid.iloc[i:i+1,4:]),np.ravel(Hormonizom.iloc[j:j+1,4:])),2)         

HormonizomSimilaity.to_csv("/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/Similarity(Jacurd)_Between_Hormonizom_CovidDrugCandid.csv",index=False,sep="\t")        
 

##### Calculate Our Similarity for Hormonizom  ##############
####################################################
count=0
HormonizomSimilaity=pd.DataFrame(columns=['Code1','CID1','ClassCode1','Code2','CID2','ClassCode2','Similarity'])
# i=0
for i in Hormonizom_Candid.index:
    for j in Hormonizom.index:
        count=count+1
        print(str(count))
        CID1=Hormonizom_Candid.at[i,'CID']
        CID2=Hormonizom.at[j,'CID']
        Code1=Hormonizom_Candid.at[i,'Code']
        Code2=Hormonizom.at[j,'Code']        
        row=len(HormonizomSimilaity)
        HormonizomSimilaity.at[row,'CID1']=CID1
        HormonizomSimilaity.at[row,'CID2']=CID2
        HormonizomSimilaity.at[row,'Code1']=Code1
        HormonizomSimilaity.at[row,'Code2']=Code2        
        HormonizomSimilaity.at[row,'ClassCode1']=Hormonizom_Candid.at[i,'Class']
        HormonizomSimilaity.at[row,'ClassCode2']=Hormonizom.at[j,'Class']
        HormonizomSimilaity.at[row,'Similarity']=round(HormonizomSimilarity(np.ravel(Hormonizom_Candid.iloc[i:i+1,4:]),np.ravel(Hormonizom.iloc[j:j+1,4:]),len(Hormonizom.columns[4:])),2)         

HormonizomSimilaity.to_csv("/Users/aghil/Desktop/uni/Tese 2/Implementation/Result/Hormonizom_Chemical/solution_final/Similarity(Jacurd)_Between_Hormonizom_CovidDrugCandid.csv",index=False,sep="\t")        
 




#score for Hormonizom similarity
################################
def HormonizomSimilarity(list1, list2,AllFeatureCount):
    Score=0
    for i in range(AllFeatureCount):
      if list1[i]==0 and list2[i]==0  :
          Score=Score+0
      elif (list1[i] == list2[i]) and (list1[i]!=0):
          Score=Score+2
      elif (list1[i]==+1 and list2[i]==-1):
          Score=Score-2
      elif (list1[i]==0 and list2[i]==-1) or (list1[i]==-1 and list2[i]==0):
          Score=Score-1
      elif (list1[i]==0 and list2[i]==+1) or (list1[i]==+1 and list2[i]==0):
          Score=Score-1        
    return Score

# HormonizomSimilaity[HormonizomSimilaity.CID1==3117].iloc[:,1:].sort_values('Similarity', ascending=False)

 
######################################################
######################################################
AllClass=pd.read_csv("E:/DATA/UniTese/Data/Tese 2/All Drug_By Mesh _Class/All Drug_In Mesh_Thraputic_By_ClassCodeName.csv","\t")
AllClass=AllClass[['ClassCode','ClassName']].drop_duplicates()

All_Tonimoto=pd.read_csv("E:/DATA/UniTese/Data/Tese 2/Implementation/Result/Chemical/Similarity(tonimoto)_Between_Chemical.csv","\t")
All_Drug_In_HC_Not_In_C=pd.read_csv("E:/DATA/UniTese/Data/Tese 2/Implementation/Result/Chemical/Macc(3712-156)_L12_L12_4_classification_Remove_Macc_Col_Lower_Var0/Classification Result/diff_Is_In_HormonizomChemical_Not_In_Chemical.csv","\t")
All_Drug_In_HC_Not_In_C.insert(4,'Similarity_To_RealClass',np.float())
All_Drug_In_HC_Not_In_C.insert(5,'Similarity_To_PredictClass',np.float())

AllClass=All_Tonimoto['ClassCode2'].drop_duplicates()

for row in All_Drug_In_HC_Not_In_C.index:
   All_Drug_In_HC_Not_In_C.at[row,'Similarity_To_RealClass']=SimilarityToClass(All_Drug_In_HC_Not_In_C.at[row,'CID'],All_Drug_In_HC_Not_In_C.at[row,'RealClass'])
   All_Drug_In_HC_Not_In_C.at[row,'Similarity_To_PredictClass']=SimilarityToClass(All_Drug_In_HC_Not_In_C.at[row,'CID'],All_Drug_In_HC_Not_In_C.at[row,'PredictClass'])

All_Drug_In_HC_Not_In_C.to_csv("E:/DATA/UniTese/Data/Tese 2/Implementation/Result/Chemical/Macc(3712-156)_L12_L12_4_classification_Remove_Macc_Col_Lower_Var0/Classification Result/diff_Is_In_HormonizomChemical_Not_In_Chemical_By_TonimotoSim.csv",index=False,sep="\t")

def SimilarityToClass(drug,Class):
    Class=AllClass[AllClass['ClassName']==Class].reset_index(drop=True).at[0,'ClassCode']
    AllSimilarityClass=All_Tonimoto[(All_Tonimoto['ClassCode2']==Class) & (All_Tonimoto['CID1']==drug)]
    return(np.round(np.average(AllSimilarityClass['Tonimoto']),decimals=2))

