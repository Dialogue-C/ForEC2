import pickle
import xgboost as xgb
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE, r2_score

# 加载数据
data = load_boston()
X = data.data
y = data.target

# 划分数据
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=0)

# 设定参数，对模型进行训练
dtrain = xgb.DMatrix(Xtrain, Ytrain)
param = {'silent': True
    , 'obj': 'reg:linear'
    , "subsample": 1
    , "eta": 0.05
    , "gamma": 20
    , "lambda": 3.5
    , "alpha": 0.2
    , "max_depth": 4
    , "colsample_bytree": 0.4
    , "colsample_bylevel": 0.6
    , "colsample_bynode": 1}
num_round = 180

bst = xgb.train(param, dtrain, num_round)

# 保存模型
pickle.dump(bst, open("model_1.dat", "wb"))

# 注意，如果我们保存的模型是xgboost库中建立的模型，则导入的数据类型也必须是xgboost库中的数据类型
dtest = xgb.DMatrix(Xtest, Ytest)

# 导入模型
loaded_model = pickle.load(open("model_1.dat", "rb"))
print("Loaded model from: model_1.dat")

# 模型预测
ypreds = loaded_model.predict(dtest)

print(MSE(Ytest, ypreds))

print(r2_score(Ytest, ypreds))