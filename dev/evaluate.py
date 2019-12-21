#%%
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
#%%
# y_pred是预测标签
y_pred, y_true =[1,2,2, 2,1,2,2, 3,3], [1,1,1,2,2,2,2,3,3]
print(classification_report(y_true=y_true, y_pred=y_pred))
p = precision_score(y_true=y_true, y_pred=y_pred, average='macro')
r = recall_score(y_true=y_true, y_pred=y_pred, average='macro')
f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
acc = accuracy_score(y_true=y_true, y_pred=y_pred)
cm = confusion_matrix(y_true=y_true, y_pred=y_pred)