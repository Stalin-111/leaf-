import argparse, json
from pathlib import Path
import numpy as np, tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt, seaborn as sns

def main():
 p=argparse.ArgumentParser(); p.add_argument('--data',default='dataset/processed'); p.add_argument('--model',default='models/deepleaf_cnn.keras'); a=p.parse_args()
 ds=tf.keras.utils.image_dataset_from_directory(Path(a.data)/'test',image_size=(224,224),batch_size=32,label_mode='categorical',shuffle=False)
 model=tf.keras.models.load_model(a.model); y=[]; pred=[]
 for x,t in ds: y.extend(np.argmax(t.numpy(),1)); pred.extend(np.argmax(model.predict(x,verbose=0),1))
 names=ds.class_names; Path('reports').mkdir(exist_ok=True)
 report=classification_report(y,pred,target_names=names,digits=4); Path('reports/classification_report.txt').write_text(report)
 cm=confusion_matrix(y,pred); plt.figure(figsize=(10,8)); sns.heatmap(cm,annot=True,fmt='d',xticklabels=names,yticklabels=names,cbar=False); plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.tight_layout(); plt.savefig('reports/confusion_matrix.png',dpi=160); plt.close(); print(report)
if __name__=='__main__': main()
