import pandas as pd
import numpy as np

########### Tonimoto Coefficient ##########
###########################################
# Inputs: two lists
# Output: the Tanimoto Coefficient
def tanimoto (list1, list2):
    c=np.count_nonzero(np.multiply(list1,list2)==1)
    return c/(np.count_nonzero(list1)+np.count_nonzero(list2)-c)

Hormonizom=pd.read_csv("E:/DATA/UniTese/Data/Tese 2/final list/Train_By Common Multimodal Data/only hormonizom/each Cell Line Dose And Time In One Row/All_Hormonizom_Col_Lower_VarQuantile0.75_Remove.csv",delimiter="\t")
Hormonizom=Hormonizom.dropna()
MaccData=pd.read_csv("E:/DATA/UniTese/Data/Tese 2/final list/Train_By Common Multimodal Data/only hormonizom/each Cell Line Dose And Time In One Row/All_Macc_FP_Col_Var0_Remove.csv",delimiter="\t")
MaccData.rename(columns={'Name': 'CID'}, inplace=True)
MaccData.insert(1,'Class',Hormonizom['Class'])
MaccData=MaccData.drop_duplicates()
MaccData=MaccData[MaccData['Class']!='NoClass']
MaccData=MaccData.reset_index(drop=True)

##### Calculate tonimoto for Chemical ##############
####################################################
count=0
ChemicalTonimotoSimilaity=pd.DataFrame(columns=['CID1','ClassCode1','CID2','ClassCode2','Tonimoto'])
# i=0
for i in MaccData.index:
    for j in MaccData.index:
        count=count+1
        print(str(count))
        CID1=MaccData.at[i,'CID']
        CID2=MaccData.at[j,'CID']
        row=len(ChemicalTonimotoSimilaity)
        ChemicalTonimotoSimilaity.at[row,'CID1']=CID1
        ChemicalTonimotoSimilaity.at[row,'CID2']=CID2
        ChemicalTonimotoSimilaity.at[row,'ClassCode1']=MaccData.at[i,'Class']
        ChemicalTonimotoSimilaity.at[row,'ClassCode2']=MaccData.at[j,'Class']
        ChemicalTonimotoSimilaity.at[row,'Tonimoto']=round(tanimoto(MaccData.iloc[i:i+1,2:],MaccData.iloc[j:j+1,2:]),2)         
ChemicalTonimotoSimilaity.to_csv("E:/DATA/UniTese/Data/Tese 2/Implementation/Result/Chemical/Similarity(tonimoto)_Between_Chemical.csv",index=False,sep="\t")        
        
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

