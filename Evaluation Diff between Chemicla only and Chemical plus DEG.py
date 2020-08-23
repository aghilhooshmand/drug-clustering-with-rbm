import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


diff_In_Chem_Not_In_DEG=pd.read_csv('E:/DATA/UniTese/Data/Tese 2/Implementation/Result/Chemical/Macc(29074-156)_L12_L12_4_classification_Remove_Macc_Col_Lower_Var0/Classification Result/For Report/diff_Is_In_Chemical_Not_In_HormonizomChemical.csv',delimiter="\t")
diff_In_Chem_Not_In_DEG=diff_In_Chem_Not_In_DEG.drop_duplicates()
diff_In_Chem_Not_In_DEG=diff_In_Chem_Not_In_DEG.reset_index(drop=True)
diff_In_DEG_Not_In_Chem=pd.read_csv('E:/DATA/UniTese/Data/Tese 2/Implementation/Result/Chemical/Macc(29074-156)_L12_L12_4_classification_Remove_Macc_Col_Lower_Var0/Classification Result/For Report/diff_Is_In_HormonizomChemical_Not_In_Chemical.csv',delimiter="\t")
diff_In_DEG_Not_In_Chem=diff_In_DEG_Not_In_Chem.drop_duplicates()
Similarity_Matrix=pd.read_csv('E:/DATA/UniTese/Data/Tese 2/Implementation/Result/Chemical/Similarity(tonimoto)_Between_Chemical.csv',delimiter="\t")
diff_Chem_With_All=Similarity_Matrix[Similarity_Matrix.CID1.isin(diff_In_Chem_Not_In_DEG.CID)]
diff_DEG_With_All=Similarity_Matrix[Similarity_Matrix.CID1.isin(diff_In_DEG_Not_In_Chem.CID)]

diff_Chem_With_All_final=diff_Chem_With_All.drop(diff_Chem_With_All[diff_Chem_With_All.CID1==diff_Chem_With_All.CID2].index,axis=0)
diff_Chem_With_All_final=diff_Chem_With_All_final.drop(diff_Chem_With_All_final[diff_Chem_With_All_final.ClassCode1.isin([11,12,16,4,17])].index,axis=0)
diff_Chem_With_All_final=diff_Chem_With_All_final.drop(diff_Chem_With_All_final[diff_Chem_With_All_final.ClassCode2.isin([11,12,16,4,17])].index,axis=0)

diff_DEG_With_All_final=diff_DEG_With_All.drop(diff_DEG_With_All[diff_DEG_With_All.CID1==diff_DEG_With_All.CID2].index,axis=0)
diff_DEG_With_All_final=diff_DEG_With_All_final.drop(diff_DEG_With_All_final[diff_DEG_With_All_final.ClassCode1.isin([11,12,16,4,17])].index,axis=0)
diff_DEG_With_All_final=diff_DEG_With_All_final.drop(diff_DEG_With_All_final[diff_DEG_With_All_final.ClassCode2.isin([11,12,16,4,17])].index,axis=0)



AllClass=pd.read_csv("E:/DATA/UniTese/Data/Tese 2/All Drug_By Mesh _Class/All Drug_In Mesh_Thraputic_By_ClassCodeName.csv",delimiter="\t")
AllClass=AllClass[['ClassCode','ClassName']].drop_duplicates().reset_index(drop=True)

diff_Chem_With_All_final['ClassCode1'] = diff_Chem_With_All_final['ClassCode1'].map({1:AllClass.loc[0,'ClassName'],2:AllClass.loc[1,'ClassName'],3:AllClass.loc[2,'ClassName']
            ,4:AllClass.loc[3,'ClassName'],5:AllClass.loc[4,'ClassName'],6:AllClass.loc[5,'ClassName']
            ,7:AllClass.loc[6,'ClassName'],8:AllClass.loc[7,'ClassName'],9:AllClass.loc[8,'ClassName']
            ,10:AllClass.loc[9,'ClassName'],11:AllClass.loc[10,'ClassName'],12:AllClass.loc[11,'ClassName']
            ,13:AllClass.loc[12,'ClassName'],14:AllClass.loc[13,'ClassName'],15:AllClass.loc[14,'ClassName']
            ,16:AllClass.loc[15,'ClassName']})
diff_Chem_With_All_final['ClassCode2'] = diff_Chem_With_All_final['ClassCode2'].map({1:AllClass.loc[0,'ClassName'],2:AllClass.loc[1,'ClassName'],3:AllClass.loc[2,'ClassName']
            ,4:AllClass.loc[3,'ClassName'],5:AllClass.loc[4,'ClassName'],6:AllClass.loc[5,'ClassName']
            ,7:AllClass.loc[6,'ClassName'],8:AllClass.loc[7,'ClassName'],9:AllClass.loc[8,'ClassName']
            ,10:AllClass.loc[9,'ClassName'],11:AllClass.loc[10,'ClassName'],12:AllClass.loc[11,'ClassName']
            ,13:AllClass.loc[12,'ClassName'],14:AllClass.loc[13,'ClassName'],15:AllClass.loc[14,'ClassName']
            ,16:AllClass.loc[15,'ClassName']})

diff_DEG_With_All_final['ClassCode1'] = diff_DEG_With_All_final['ClassCode1'].map({1:AllClass.loc[0,'ClassName'],2:AllClass.loc[1,'ClassName'],3:AllClass.loc[2,'ClassName']
            ,4:AllClass.loc[3,'ClassName'],5:AllClass.loc[4,'ClassName'],6:AllClass.loc[5,'ClassName']
            ,7:AllClass.loc[6,'ClassName'],8:AllClass.loc[7,'ClassName'],9:AllClass.loc[8,'ClassName']
            ,10:AllClass.loc[9,'ClassName'],11:AllClass.loc[10,'ClassName'],12:AllClass.loc[11,'ClassName']
            ,13:AllClass.loc[12,'ClassName'],14:AllClass.loc[13,'ClassName'],15:AllClass.loc[14,'ClassName']
            ,16:AllClass.loc[15,'ClassName']})
diff_DEG_With_All_final['ClassCode2'] = diff_DEG_With_All_final['ClassCode2'].map({1:AllClass.loc[0,'ClassName'],2:AllClass.loc[1,'ClassName'],3:AllClass.loc[2,'ClassName']
            ,4:AllClass.loc[3,'ClassName'],5:AllClass.loc[4,'ClassName'],6:AllClass.loc[5,'ClassName']
            ,7:AllClass.loc[6,'ClassName'],8:AllClass.loc[7,'ClassName'],9:AllClass.loc[8,'ClassName']
            ,10:AllClass.loc[9,'ClassName'],11:AllClass.loc[10,'ClassName'],12:AllClass.loc[11,'ClassName']
            ,13:AllClass.loc[12,'ClassName'],14:AllClass.loc[13,'ClassName'],15:AllClass.loc[14,'ClassName']
            ,16:AllClass.loc[15,'ClassName']})




pivot=diff_Chem_With_All_final.groupby(['CID1', 'ClassCode2'])['Tonimoto'].aggregate('mean').round(3).unstack()
plt.subplots(figsize=(40,40))
sns.set(font_scale=4)
ax=sns.heatmap(pivot,cmap="YlGnBu",annot=True)
ax.set(ylabel='Compund ID', xlabel='Mesh teraputic class')
from matplotlib.patches import Rectangle
for i in range(len(diff_In_Chem_Not_In_DEG)):
    ax.add_patch(Rectangle((FindLocInPivot(diff_In_Chem_Not_In_DEG.at[i,'CID'],diff_In_Chem_Not_In_DEG.at[i,'PredictClassName'])), 1, 1, fill=False, edgecolor='red', lw=4))
plt.show()
pd.DataFrame(pivot).to_csv("E:/DATA/UniTese/Data/Tese 2/Implementation/Result/Chemical/Macc(29074-156)_L12_L12_4_classification_Remove_Macc_Col_Lower_Var0/Classification Result/For Report/Pivot_diff_Is_In_Chemical_Not_In_HormonizomChemical_MeanTonimoto_to_OtherClass.csv",sep="\t",index=True)

pivot=diff_DEG_With_All_final.groupby(['CID1', 'ClassCode2'])['Tonimoto'].aggregate('mean').round(3).unstack()
plt.subplots(figsize=(50,60))
sns.set(font_scale=4)
ax=sns.heatmap(pivot,cmap="YlGnBu",annot=True)
ax.set(ylabel='Compund ID', xlabel='Mesh teraputic class')
from matplotlib.patches import Rectangle
for i in range(len(diff_In_DEG_Not_In_Chem)):
    ax.add_patch(Rectangle((FindLocInPivot(diff_In_DEG_Not_In_Chem.at[i,'CID'],diff_In_DEG_Not_In_Chem.at[i,'PredictClass'])), 1, 1, fill=False, edgecolor='red', lw=4))
plt.show()
pd.DataFrame(pivot).to_csv("E:/DATA/UniTese/Data/Tese 2/Implementation/Result/Chemical/Macc(29074-156)_L12_L12_4_classification_Remove_Macc_Col_Lower_Var0/Classification Result/For Report/Pivot_diff_Is_In_HormonizomChemical_Not_In_Chemical_MeanTonimoto_to_OtherClass.csv",sep="\t",index=True)




#Function For Find Location CID and Class in Heatmap
def FindLocInPivot(CID,Class):
    for i in range(len(pivot)):
        if pivot.index[(i)]==CID:
            X=i
    for i in range(len(pivot.columns)):
        if pivot.columns[(i)]==Class:
            Y=i
    return Y,X
    
    
    


    