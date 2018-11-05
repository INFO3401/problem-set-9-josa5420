"""from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
import pandas as pd
class AnalysisData:
    def __init__(self, dataset):
        self.dataset = dataset   #Which holds the parsed dataset
        self.variables = [i for i in self.dataset.columns if i != 'competitorname']#.remove('competitorname') #which will hold a list containing the indexes for all of the variables in your data
        self.targetY = ""
    
    def setTargetY(self, a):
        self.targetY = a
    
class LinearAnalysis: #will contain your functions for doing linear regression
    def __init__(self, targetY):
        self.bestX = 0#bestX #holds the best X predictor for data
        self.targetY = targetY#targetY #holds the index to the target dependent variable
        self.fit = 0 #fit  #will hold how well bestX predicts target variable
    def runSimpleAnalysis(self,t):
        acc = 0
        var = ""
        test = [i for i in t.variables if i != self.targetY]
        for i in test:
            regr = LinearRegression()
            regr.fit(t.dataset[i].reshape(-1,1),t.dataset[self.targetY].reshape(-1,1))
            predict = regr.predict(t.dataset[i].reshape(-1,1))
            a = r2_score(t.dataset[self.targetY].reshape(-1,1), predict)
            print(a)
            if a > acc:
                acc = a
                var = i
        print(str(var)+": "+str(acc))
        
    
class LogisticAnalysis: #will contain your functions for doing logistic regression
    def __init__(self, targetY):
        self.bestX = 0#bestX #holds the best X predictor for data
        self.targetY = targetY#targetY #holds the index to the target dependent variable
        self.fit = 0#fit  #will hold how well bestX predicts target variable
    def runSimpleAnalysis(self,t):
        acc = 0
        var = ""
        test = [i for i in t.variables if i != self.targetY]
        for i in test:
            regr = LogisticRegression()
            regr.fit(t.dataset[i].astype(int).reshape(-1,1),t.dataset[self.targetY].astype(int).reshape(-1,1))
            predict = regr.predict(t.dataset[i].reshape(-1,1))
            a = r2_score(t.dataset[self.targetY].reshape(-1,1), predict)
            if a > acc:
                acc = a
                var = i
        print(str(var)+": "+str(acc))
        
df = pd.read_csv('candy-data.csv')
AD1 = AnalysisData(df)
AD1.setTargetY('sugarpercent')
print(AD1.targetY)

LA1 = LinearAnalysis(AD1.targetY)
LA1.runSimpleAnalysis(AD1)        
"""