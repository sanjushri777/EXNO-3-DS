## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
<img width="265" height="331" alt="image" src="https://github.com/user-attachments/assets/3f828a2e-53b8-4875-8bd6-27f274022e9d" />

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
<img width="216" height="177" alt="image" src="https://github.com/user-attachments/assets/f5a7137c-dd65-4897-bcee-1e7009da0d26" />

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
<img width="381" height="338" alt="image" src="https://github.com/user-attachments/assets/046f59d7-f19e-4024-bc0a-178a3c148cc0" />

```
 le=LabelEncoder()
 dfc=df.copy()
 dfc['ord_2']=le.fit_transform(dfc['ord_2'])
 dfc
```

<img width="446" height="437" alt="image" src="https://github.com/user-attachments/assets/d5baac9d-62f0-4859-aeb3-4c02a7234c43" />

```
 from sklearn.preprocessing import OneHotEncoder
 ohe=OneHotEncoder()
 df2=df.copy()
 enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
 df2=pd.concat([df2,enc],axis=1)
 df2
```
<img width="697" height="339" alt="image" src="https://github.com/user-attachments/assets/f7115f24-b1c3-4c89-84b3-1aa6dd2a3788" />

```
 pd.get_dummies(df2,columns=["nom_0"])

```
<img width="836" height="491" alt="image" src="https://github.com/user-attachments/assets/9ea05ec1-2231-435b-8dda-7d0203b10458" />

```
 pip install--upgrade category_encoders

```
<img width="839" height="322" alt="image" src="https://github.com/user-attachments/assets/ed8ef479-5eb1-49c9-b3ee-31dc82785db7" />

```

 from category_encoders import BinaryEncoder
 df=pd.read_csv("data.csv")
 df
 be=BinaryEncoder()
 nd=be.fit_transform(df['Ord_2'])
 df
 dfb=pd.concat([df,nd],axis=1)
 dfb
```
<img width="689" height="339" alt="image" src="https://github.com/user-attachments/assets/78f994ed-361b-4bea-a4da-87653c6de4ae" />

```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
<img width="538" height="330" alt="image" src="https://github.com/user-attachments/assets/8dac4fa9-579f-4d8a-8aa0-94ad8e958c02" />

```
 import pandas as pd
 from scipy import stats
 import numpy as np
 df=pd.read_csv("Data_to_Transform.csv")
 df
```
<img width="793" height="405" alt="image" src="https://github.com/user-attachments/assets/512d3683-a048-475e-ba66-3c0f3868ea24" />

```
df.skew()

```

<img width="482" height="283" alt="image" src="https://github.com/user-attachments/assets/38efaf05-2a84-480b-b24f-d9c27b6263b0" />

```
 np.log(df["Highly Positive Skew"])

```
<img width="425" height="605" alt="image" src="https://github.com/user-attachments/assets/2325e95b-77ba-4e7a-9954-8cb28af224f0" />

```
np.reciprocal(df["Moderate Positive Skew"])

```
<img width="452" height="607" alt="image" src="https://github.com/user-attachments/assets/bd65b308-c442-4391-8307-df3c252936a8" />

```
 np.sqrt(df["Highly Positive Skew"])

```
<img width="403" height="606" alt="image" src="https://github.com/user-attachments/assets/e5e737ea-7686-4efd-8954-ea84db1f14dd" />

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
 df
```
<img width="922" height="452" alt="image" src="https://github.com/user-attachments/assets/e355cbb6-a613-4219-82c5-79765b35ef21" />

```
 df.skew()

```
<img width="342" height="215" alt="image" src="https://github.com/user-attachments/assets/3e9e6e3a-5e38-400b-8d20-add4aad0708f" />

```
 df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
 df.skew()

```
<img width="368" height="242" alt="image" src="https://github.com/user-attachments/assets/71565917-5ba6-4a74-bdd2-6ae49a93bbf3" />

```
 from sklearn.preprocessing import QuantileTransformer
 qt=QuantileTransformer(output_distribution='normal')
 df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
 df

```
<img width="643" height="481" alt="image" src="https://github.com/user-attachments/assets/d408c682-89ef-43e4-ade9-1ad92222cb96" />

```
 import seaborn as sns
 import statsmodels.api as sm
 import matplotlib.pyplot as plt
 sm.qqplot(df["Moderate Negative Skew"],line='45')
 plt.show()

```
<img width="621" height="459" alt="image" src="https://github.com/user-attachments/assets/2154dd94-af31-40e3-8672-1c50342cd8b7" />

```
 sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
 plt.show()

```
<img width="621" height="454" alt="image" src="https://github.com/user-attachments/assets/8a94e646-6440-4b61-b5d4-fce4db5d9e24" />

```
 from sklearn.preprocessing import QuantileTransformer
 qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
 df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
 sm.qqplot(df["Moderate Negative Skew"],line='45')
 plt.show()

```
<img width="622" height="456" alt="image" src="https://github.com/user-attachments/assets/d3a3cbb1-c77f-486f-b20d-cb8df15fe20a" />

```
 df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
 sm.qqplot(df["Highly Negative Skew"],line='45')
 plt.show()

```
<img width="639" height="458" alt="image" src="https://github.com/user-attachments/assets/b1cd0e6d-cb15-405a-9794-31520cc6fa45" />

```
dt=pd.read_csv("titanic_dataset.csv")
 dt
 from sklearn.preprocessing import QuantileTransformer
 qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
 dt["Age_1"]=qt.fit_transform(dt[["Age"]])
 sm.qqplot(dt['Age'],line='45') 
plt.show()

```
<img width="571" height="416" alt="image" src="https://github.com/user-attachments/assets/cf6a2828-80c6-4d24-aacc-355e08da0b09" />

```
 sm.qqplot(df["Highly Negative Skew_1"],line='45')
 plt.show()

```
<img width="567" height="416" alt="image" src="https://github.com/user-attachments/assets/a7df6c0b-243b-4022-a9e6-dea64ccfc1de" />













































      
# RESULT:
 Thus the given data, Feature Encoding, Transformation process and save the data to a file
 was performed successfully.
       

       
