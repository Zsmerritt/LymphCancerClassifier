import pandas as pd, shutil

#0 = No Cancer
#1 = Cancer
dataset = pd.read_csv('train_labels.csv')
names = list(dataset['id'].values)
labels = list(dataset['label'].values)

for x, name in enumerate(names):
	label='cancer' if labels[x]==1 else 'noCancer'
	shutil.move('./data/train/'+name+'.tif','./data/train/'+label+'/')

