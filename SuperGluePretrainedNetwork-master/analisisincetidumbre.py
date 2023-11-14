from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import cv2
from models.utils import (pose_auc)
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

print('Procedidendo a evaluar')
# Collate the results into a final table and print to terminal.
AUC3=[]
AUC5=[]
AUC10=[]
Prec=[]
Rep=[]
MScore=[]
pose_errors = []
precisions = []
matching_scores = []
matches_Correct=[]
N2=[]
matches=[]
error_t=[]
error_R=[]
for j in range(1,50):
    carpeta='EnchancedOri_'+str(j)+'.mp4_eval/'
    AUC3t=[]
    AUC5t=[]
    AUC10t=[]
    Prect=[]
    Rept=[]
    MScoret=[]
    pose_errorst = []
    precisionst = []
    matching_scorest = []
    matches_Correctt=[]
    N2=[]
    matchest=[]
    error_tt=[]
    error_Rt=[]
    thresholds = [3, 5, 10]
    for i in range(0,718):
        name0=str(i)
        name1=str(i+1)
        stem0 = Path(name0).stem
        stem1 = Path(name1).stem
        eval_path = carpeta+'{}_{}_evaluation.npz'.format(stem0,stem1)
        results = np.load(eval_path)
        error_Rt.append(results['error_R'])
        error_tt.append(results['error_t'])
        pose_error = np.maximum(results['error_t'], results['error_R'])
        pose_errorst.append(pose_error)
        precisionst.append(results['precision'])
        matching_scorest.append(results['matching_score'])
        matches_Correctt.append(results['matches_correct'])
        matchest.append(results['matches'])
    aucs = pose_auc(pose_errorst, thresholds)
    aucs = [100.*yy for yy in aucs]
    AUC3.append(aucs[0])
    AUC5.append(aucs[1])
    AUC10.append(aucs[2])
    Prec.append(100.*np.mean(precisionst))
    Rep.append(100.*(np.mean(matches_Correctt)/np.mean(matchest)))
    MScore.append(100.*np.mean(matching_scorest))
    pose_errors.append(np.mean(pose_errorst))
    precisions.append(np.mean(precisionst))
    matches_Correct.append(np.mean(matches_Correctt))
    matches.append(np.mean(matchest))
    error_t.append(np.mean(error_tt))
    error_R.append(np.mean(error_Rt))

print("Valor Maximo AUC3 "+str(max(AUC3))+', el valor minimo es de '+str(min(AUC3))+' y la diferencia de incertidumbre '+str(max(AUC3)-min(AUC3)))
print("Valor Maximo AUC5 "+str(max(AUC5))+', el valor minimo es de '+str(min(AUC5))+' y la diferencia de incertidumbre '+str(max(AUC5)-min(AUC5)))
print("Valor Maximo AUC10 "+str(max(AUC10))+', el valor minimo es de '+str(min(AUC10))+' y la diferencia de incertidumbre '+str(max(AUC10)-min(AUC10)))
print("Valor Maximo precisions "+str(max(Prec))+', el valor minimo es de '+str(min(precisions))+' y la diferencia de incertidumbre '+str(max(precisions)-min(precisions)))
print("Valor Maximo Rep "+str(max(Rep))+', el valor minimo es de '+str(min(Rep))+' y la diferencia de incertidumbre '+str(max(Rep)-min(Rep)))
print("Valor Maximo MScore "+str(max(MScore))+', el valor minimo es de '+str(min(MScore))+' y la diferencia de incertidumbre '+str(max(MScore)-min(MScore)))
print("Valor Maximo matches_Correct "+str(max(matches_Correct))+', el valor minimo es de '+str(min(matches_Correct))+' y la diferencia de incertidumbre '+str(max(matches_Correct)-min(matches_Correct)))
print("Valor Maximo matches "+str(max(matches))+', el valor minimo es de '+str(min(matches))+' y la diferencia de incertidumbre '+str(max(matches)-min(matches)))




print('AUC@3\t AUC@5\t AUC@10\t Prec\t Rep\t MScore\t')
print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
    np.mean(AUC3), np.mean(AUC5), np.mean(AUC10), np.mean(Prec),np.mean(Rep), np.mean(MScore)))
# Graficar PoseError
# Crear subgráficas
fig, axs = plt.subplots(2, 3, figsize=(30, 24))
iteraciones = np.arange(0, 49) # Números del 1 al 6 representando las 6 gráficas
plt.figure(figsize=(10, 6))
plt.subplot(2, 3, 1)
plt.plot(iteraciones, AUC3, marker='o', linestyle='-')
plt.title('AUC3')
plt.xlabel('Iteración')
plt.ylabel('Valor (%)')

# Graficar Precision
plt.subplot(2, 3, 2)
plt.plot(iteraciones, AUC5, marker='o', linestyle='-')
plt.title('AUC5')
plt.xlabel('Iteración')
plt.ylabel('Valor (%)')

# Graficar Matching Score
plt.subplot(2, 3, 3)
plt.plot(iteraciones, AUC10, marker='o', linestyle='-')
plt.title('AUC10')
plt.xlabel('Iteración')
plt.ylabel('Valor (%)')

# Graficar Matches Correctos
plt.subplot(2, 3, 4)
plt.plot(iteraciones, Prec, marker='o', linestyle='-')
plt.title('Precision')
plt.xlabel('Iteración')
plt.ylabel('Valor (%)')

# Graficar Matches Totales
plt.subplot(2, 3, 5)
plt.plot(iteraciones, Rep, marker='o', linestyle='-')
plt.title('Rep')
plt.xlabel('Iteración')
plt.ylabel('Valor (%)')

# Graficar Rep
plt.subplot(2, 3, 6)
plt.plot(iteraciones, MScore, marker='o', linestyle='-')
plt.title('MScore')
plt.xlabel('Iteración')
plt.ylabel('Valor (%)')

plt.tight_layout()
plt.show()
