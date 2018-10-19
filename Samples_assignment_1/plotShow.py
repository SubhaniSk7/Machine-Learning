import numpy as np
import sklearn as sk
import pandas as pd
import matplotlib.pyplot as mt
import xlrd
import math


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# boston_dataset=pd.read_excel('https://1.salford-systems.com/hs-fs/hub/160602/file-19987397-xls/tutorial%20datasets/boston.xls/tutorial%20datasets/boston.xls')
boston_dataset=pd.read_excel('/home/subhani007/Desktop/ML Assignment/boston.xls')

print(boston_dataset.head())

print(boston_dataset.drop('MV',axis=1))
print(boston_dataset.keys())

mt.figure(figsize=(7, 5))
# mt.show(boston_dataset.get(0))

mt.plot(boston_dataset.__getattr__('MV'))
mt.title('Linear Regression')
mt.ylabel('y')
mt.xlabel('input')
mt.show()