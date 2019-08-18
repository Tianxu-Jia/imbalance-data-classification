
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imxgboost.imbalance_xgb import imbalance_xgboost as imb_xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

import torch.nn as nn
import torch
import torch.optim as optim


def freq_encode(pd_data, columns=False):
    '''Returns a DataFrame with encoded columns'''
    if columns==False:
        columns=list(pd_data.columns)
    encoded_cols = []
    nsamples = pd_data.shape[0]
    for col in columns:
        freqs_cat = pd_data.groupby(col)[col].count()/nsamples
        encoded_col = pd_data[col].map(freqs_cat)
        encoded_col[encoded_col.isnull()] = np.nan
        encoded_cols.append(pd.DataFrame({'freq_'+str(col):encoded_col}))
    all_encoded = pd.concat(encoded_cols, axis=1)
    return all_encoded.loc[pd_data.index,:]


data_ =  pd.read_csv("data.csv")
data = data_.drop_duplicates()
data = data.replace('?', np.nan)

xx = data.iloc[:,:data.shape[1]-1]
yy = data.iloc[:,-1].values
continuous_var = [0, 5, 16, 17, 18, 29, 38]
cat_var = list(set(np.arange(40)).difference(set(continuous_var)))
cat_columns = [xx.columns[i] for i in cat_var]
cat_encoded = freq_encode(xx, cat_columns)
contns = xx.iloc[:, continuous_var]
encoded_x = np.concatenate((cat_encoded.values, contns.values), axis=1)
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(yy)
encoded_y = label_encoder.transform(yy)


seed = 7
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(encoded_x, encoded_y, test_size=test_size, random_state=seed)

xgboster_weight = imb_xgb(special_objective='weighted')
CV_weight_booster = GridSearchCV(xgboster_weight,
                                 {"imbalance_alpha":[0.001, 0.1,0.5,1.0, 1.5,2.0,2.5,3.0,4.0,8.0,26,32]})

CV_weight_booster.fit(X_train, y_train)
opt_weight_booster = CV_weight_booster.best_estimator_

class_output = opt_weight_booster.predict_determine(X_train, y=None)
imb_confusion = confusion_matrix(y_train, class_output)

precision = (imb_confusion[0,0] + imb_confusion[1,1])/len(y_train)

print("Prob(positive on the condition positive) = %.4f"%(imb_confusion[0,0]/(imb_confusion[0,0]+imb_confusion[0,1])))
print("Prob(negative on the condition negative) = %.4f"%(imb_confusion[1,1]/(imb_confusion[1,1]+imb_confusion[1,0])))

########################################
########################################
# normal XGBoost classifier with balanced training dataset
########################################
########################################

x_positive = X_train[y_train>0,:]
x_negative = X_train[y_train==0,:]
xy_positive = np.append(x_positive, np.ones((x_positive.shape[0],1)), axis=1)
xy_negative = np.append(x_negative, np.zeros((x_negative.shape[0],1)), axis=1)

neg_idx = np.random.choice(xy_negative.shape[0], xy_positive.shape[0], replace=False)
balanced_neg = xy_negative[neg_idx,:]

xy_train_ = np.vstack((xy_positive, balanced_neg))
xy_train = np.random.permutation(xy_train_)

X_train = xy_train[:, :(xy_train.shape[1]-1)]
y_train = xy_train[:,-1]

import xgboost as xgb
gbm = xgb.XGBClassifier(max_depth=5, n_estimators=300, learning_rate=0.05)
gbm.fit(X_train,y_train)

preds = gbm.predict(X_train)
proba_preds = gbm.predict_proba(X_train)
proba_output = opt_weight_booster.predict_sigmoid(X_train, y=None)

confusion = confusion_matrix(y_train, preds)

print("Prob(positive on the condition positive) = %.4f"%(confusion[0,0]/(confusion[0,0]+confusion[0,1])))
print("Prob(negative on the condition negative) = %.4f"%(confusion[1,1]/(confusion[1,1]+confusion[1,0])))

xx = np.transpose(np.vstack((proba_preds[:,1], proba_output)))

class dataset(torch.utils.data.Dataset):
    def __init__(self, xx, yy):
        self.len = len(yy)
        self.xx = torch.from_numpy(xx)
        self.yy = torch.from_numpy(yy)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.xx[idx], self.yy[idx]

feature_dataset = dataset(xx, y_train)
fleature_dataloader = torch.utils.data.DataLoader(dataset=feature_dataset,
                                                  batch_size=64,
                                                  shuffle=True)

from torch.autograd import Variable

class fusion_nn(nn.Module):
    def __init__(self):
        super(fusion_nn, self).__init__()
        self.linear1 = nn.Linear(2, 50)
        self.linear2 = nn.Linear(50, 1)
        self.softmax = nn.Softmax()

    def forward(self, xx):
        f = self.softmax(self.linear1(xx))
        out = self.softmax(self.linear2(f))
        return out

class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 50)
        self.relu1 = nn.ReLU()
        self.dout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(50, 100)
        self.prelu = nn.PReLU(1)
        self.out = nn.Linear(100, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        dout = self.dout(h1)
        a2 = self.fc2(dout)
        h2 = self.prelu(a2)
        a3 = self.out(h2)
        y = self.out_act(a3)
        return y

import torchvision
#from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter()

model = Net()
criterion = nn.BCELoss()
#optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
print_loss_step = 10
total_loss = 0

for epoch in range(10):
    for i, data in enumerate(fleature_dataloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)
        y_pred = model(inputs)
        loss = criterion(y_pred, labels.to(torch.float32))

        if i%print_loss_step == 0:
            print('epoch: ', epoch, '  i: ', i, '  loss: ', total_loss/print_loss_step)
            total_loss = 0
        else:
            total_loss += loss.data

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


##########
# test
##########

im_test_out = opt_weight_booster.predict_sigmoid(X_test, y=None)
test_out = preds = gbm.predict_proba(X_test)
xx = np.transpose(np.vstack((test_out[:,1], proba_output)))

xx_input = Variable(xx)
yy_out = model(xx_input)

confusion_final = confusion_matrix(y_train, yy_out)
print(confusion_final)

debug = 1

