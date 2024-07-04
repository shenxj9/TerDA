def inrange(value, start, stop):
    return start <= value < stop


def Mw(x):
    if inrange(x, 0, 400):
        return 1
    elif inrange(x, 400, 600):
        return 2
    elif inrange(x, 600, 1000):
        return 3
    else:
        return 4


def Qed(x):
    if inrange(x, 0, 0.2):
        return 5
    elif inrange(x, 0.2, 0.4):
        return 6
    elif inrange(x, 0.4, 0.6):
        return 7
    else:
        return 8


def logP(x):
    if x < 0:
        return 9
    elif inrange(x, 0, 5):
        return 10
    elif inrange(x, 5, 8):
        return 11
    else:
        return 12


def Mr(x):
    if inrange(x, 0, 80):
        return 13
    elif inrange(x, 80, 150):
        return 14
    elif inrange(x, 150, 230):
        return 15
    else:
        return 16


def HBD(x):
    if inrange(x, 0, 3):
        return 17
    elif inrange(x, 3, 6):
        return 18
    elif inrange(x, 6, 9):
        return 19
    else:
        return 20


def HBA(x):
    if inrange(x, 0, 5):
        return 21
    elif inrange(x, 5, 10):
        return 22
    elif inrange(x, 10, 15):
        return 23
    else:
        return 24


def RB(x):
    if inrange(x, 0, 5):
        return 25
    elif inrange(x, 5, 10):
        return 26
    elif inrange(x, 10, 15):
        return 27
    else:
        return 28


def TPSA(x):
    if inrange(x, 0, 60):
        return 29
    elif inrange(x, 60, 120):
        return 30
    elif inrange(x, 120, 200):
        return 31
    else:
        return 32


def AromaticRings(x):
    if x == 0:
        return 33
    elif x == 1:
        return 34
    elif x == 2:
        return 35
    else:
        return 36


def FractionCSP3(x):
    if inrange(x, 0, 0.3):
        return 37
    elif inrange(x, 0.3, 0.6):
        return 38
    elif inrange(x, 0.3, 0.8):
        return 39
    else:
        return 40

import pandas as pd
f = r'dataset/description_values.csv'
df = pd.read_csv(f,sep = '\t')
df1 = df[['id','smiles']]
df1.head(2)

df1['Mw'] = df['Mw'].apply(lambda x: Mw(x))
df1['Qed'] = df['Qed'].apply(lambda x: Qed(x))
df1['logP'] = df['logP'].apply(lambda x: logP(x))
df1['Mr'] = df['Mr'].apply(lambda x: Mr(x))
df1['HBD'] = df['HBD'].apply(lambda x: HBD(x))
df1['HBA'] = df['HBA'].apply(lambda x: HBA(x))
df1['RB'] = df['RB'].apply(lambda x: RB(x))
df1['TPSA'] = df['TPSA'].apply(lambda x: TPSA(x))
df1['AromaticRings'] = df['AromaticRings'].apply(lambda x: AromaticRings(x))
df1['FractionCSP3'] = df['FractionCSP3'].apply(lambda x: FractionCSP3(x))
f1 = r'dataset/description.csv'
df1.to_csv(f1,sep = '\t',index = 0)
df1.head(2)