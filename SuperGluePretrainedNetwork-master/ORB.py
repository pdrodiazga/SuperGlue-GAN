import cv2
import numpy as np
from pathlib import Path
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics,AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor)

# Inicializar el detector ORB
orb = cv2.ORB_create()

# Inicializar el objeto FLANN
flann = cv2.FlannBasedMatcher()

# Capturar video desde la cámara
cap = cv2.VideoCapture('EnchancedOri1.mp4')  # Reemplaza 'ruta_de_tu_video.mp4' con la ruta de tu video

# Leer el primer frame
ret, first_frame = cap.read()

valores = [1163.45, 0., 653.626, 0., 1164.79 ,481.6, 0. ,0., 1.]
K = np.array(valores).astype(float).reshape((3, 3))
imgCont=0

# Leer el primer frame
ret, prev_frame = cap.read()

# Convertir el primer frame a escala de grises
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Encontrar los puntos clave y descriptores con ORB en el primer frame
prev_keypoints, prev_descriptors = orb.detectAndCompute(prev_gray, None)
imgCont=0
while cap.isOpened():
    # Leer un frame del video
    ret, frame = cap.read()

    if not ret:
        print("No se pudo leer el video.")
        break
    stem0, stem1 = imgCont,imgCont+1
    eval_path = "Orb_Eval/"+'{}_{}_evaluation.npz'.format(stem0, stem1)
    imgCont+=1
    # Convertir el frame actual a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Encontrar los puntos clave y descriptores con ORB en el frame actual
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    # Realizar la coincidencia de características usando FLANN
    matches = flann.knnMatch(np.float32(prev_descriptors), np.float32(descriptors), k=2)


    # Filtrar los buenos emparejamientos según el ratio de Lowe
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    mkpts0 = np.float32([prev_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    mkpts1 = np.float32([keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    mkpts0 = np.array(mkpts0)[:, 0, :]
    mkpts1 =np.array(mkpts1)[:, 0, :]
    # Dibujar los emparejamientos en el frame actual
    img_matches = cv2.drawMatches(prev_frame, prev_keypoints, frame, keypoints, good_matches, None)

    # Mostrar el resultado
    cv2.imshow('ORB with FLANN', img_matches)
    try:
        E, mask = cv2.findEssentialMat(mkpts0, mkpts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        retval, R, t, mask = cv2.recoverPose(E, mkpts0, mkpts1, K)
    except cv2.error:
        print("Error en la estimación de pose.")
        continue
    # Matriz de transformación de la cámara relativa (extrínseca)
    T_0to1 = np.eye(4)
    T_0to1[:3, :3] = R
    T_0to1[:3, 3] = t.ravel()
    epi_errs = compute_epipolar_error(mkpts0, mkpts1, T_0to1, K, K)
    correct = epi_errs < 5e-4
    num_correct = np.sum(correct)
    precision = np.mean(correct) if len(correct) > 0 else 0
    matching_score = num_correct / len(keypoints) if len(keypoints) > 0 else 0
    # Write the evaluation results to disk.
    thresh = 1.  # In pixels relative to resized image size.
    ret = estimate_pose(mkpts0, mkpts1, K, K, thresh)
    if ret is None:
        err_t, err_R = np.inf, np.inf
    else:
        R, t, inliers = ret
        err_t, err_R = compute_pose_error(T_0to1, R, t)
    out_eval = {'error_t': err_t,
                'error_R': err_R,
                'precision': precision,
                'matching_score': matching_score,
                'num_correct': num_correct,
                'epipolar_errors': epi_errs,
                'matches_correct':num_correct,
                'matches': len(matches)}
    np.savez(str(eval_path), **out_eval)
    # Actualizar el frame y los descriptores anteriores
    prev_frame = frame
    prev_gray = gray
    prev_keypoints = keypoints
    prev_descriptors = descriptors

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
print('Proceidendo a evaluar')
# Collate the results into a final table and print to terminal.
pose_errors = []
precisions = []
matching_scores = []
matches_Correct=[]
N2=[]
matches=[]
for i in range(0,imgCont):
    name0=str(i)
    name1=str(i+1)
    stem0, stem1 = Path(name0).stem, Path(name1).stem
    try:
        eval_path = "Orb_Eval/"+'{}_{}_evaluation.npz'.format(stem0, stem1)
        results = np.load(eval_path)
        pose_error = np.maximum(results['error_t'], results['error_R'])
        pose_errors.append(pose_error)
        precisions.append(results['precision'])
        matching_scores.append(results['matching_score'])
        matches_Correct.append(results['matches_correct'])
        matches.append(results['matches'])
    except:
        print("La imagen "+"Orb_Eval/"+'{}_{}_evaluation.npz'.format(stem0, stem1)+' no se guardo correctamente')

thresholds = [3, 5, 10]
aucs = pose_auc(pose_errors, thresholds)
aucs = [100.*yy for yy in aucs]
prec = 100.*np.mean(precisions)
ms = 100.*np.mean(matching_scores)
print(np.mean(matches_Correct))
print(np.mean(matches))
R=100.*(np.mean(matches_Correct)/np.mean(matches))
print('Evaluation Results (mean over {} pairs):'.format(3930))
print('AUC@3\t AUC@5\t AUC@10\t Rep\t ')
print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
    aucs[0], aucs[1], aucs[2], R))

'''
while cap.isOpened():
    # Leer un frame del video
    ret, frame = cap.read()

      # Salir del bucle si no hay más fotogramas
    if not ret:
        break
    stem0, stem1 = imgCont,imgCont+1
    eval_path = "Orb_Eval/"+'{}_{}_evaluation.npz'.format(stem0, stem1)
    imgCont+=1

    # Convertir el frame a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    keypoints1, descriptors1 = orb.detectAndCompute(first_gray, None)
    # Encontrar los puntos clave y descriptores con ORB
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    # Filtrar los puntos clave y descriptores usando un umbral de confianza
    confidence_threshold = 0.04
    filtered_keypoints = [keypoint for keypoint in keypoints if keypoint.response > confidence_threshold]
    filtered_descriptors = descriptors[[keypoint.response > confidence_threshold for keypoint in keypoints]]

    # Realizar la coincidencia de características usando FLANN
    matches = flann.knnMatch(filtered_descriptors, filtered_descriptors, k=2)

    # Filtrar los buenos emparejamientos según el ratio de Lowe
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    mkpts0 = np.float32([keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    mkpts1 = np.float32([keypoints1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    mkpts0 = np.array(mkpts0)[:, 0, :]
    mkpts1 =np.array(mkpts1)[:, 0, :]
    # Dibujar los emparejamientos en el frame
    img_matches = cv2.drawMatches(frame, filtered_keypoints, frame, filtered_keypoints, good_matches, None)

    # Mostrar el resultado
    cv2.imshow('ORB with FLANN', img_matches)

    # Salir del bucle si se presiona la tecla 'q'
    try:
        E, mask = cv2.findEssentialMat(mkpts0, mkpts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        retval, R, t, mask = cv2.recoverPose(E, mkpts0, mkpts1, K)
    except cv2.error:
        print("Error en la estimación de pose.")
        continue
    # Matriz de transformación de la cámara relativa (extrínseca)
    T_0to1 = np.eye(4)
    T_0to1[:3, :3] = R
    T_0to1[:3, 3] = t.ravel()
    epi_errs = compute_epipolar_error(mkpts0, mkpts1, T_0to1, K, K)
    correct = epi_errs < 5e-4
    num_correct = np.sum(correct)
    precision = np.mean(correct) if len(correct) > 0 else 0
    matching_score = num_correct / len(keypoints1) if len(keypoints1) > 0 else 0
    # Write the evaluation results to disk.
    thresh = 1.  # In pixels relative to resized image size.
    ret = estimate_pose(mkpts0, mkpts1, K, K, thresh)
    if ret is None:
        err_t, err_R = np.inf, np.inf
    else:
        R, t, inliers = ret
        err_t, err_R = compute_pose_error(T_0to1, R, t)
    out_eval = {'error_t': err_t,
                'error_R': err_R,
                'precision': precision,
                'matching_score': matching_score,
                'num_correct': num_correct,
                'epipolar_errors': epi_errs,
                'matches_correct':num_correct,
                'matches': len(matches)}
    np.savez(str(eval_path), **out_eval)
    first_frame=frame
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
print('Proceidendo a evaluar')
# Collate the results into a final table and print to terminal.
pose_errors = []
precisions = []
matching_scores = []
matches_Correct=[]
N2=[]
matches=[]
for i in range(0,imgCont):
    name0=str(i)
    name1=str(i+1)
    stem0, stem1 = Path(name0).stem, Path(name1).stem
    try:
        eval_path = "Orb_Eval/"+'{}_{}_evaluation.npz'.format(stem0, stem1)
        results = np.load(eval_path)
        pose_error = np.maximum(results['error_t'], results['error_R'])
        pose_errors.append(pose_error)
        precisions.append(results['precision'])
        matching_scores.append(results['matching_score'])
        matches_Correct.append(results['matches_correct'])
        matches.append(results['matches'])
    except:
        print("La imagen "+"Orb_Eval/"+'{}_{}_evaluation.npz'.format(stem0, stem1)+' no se guardo correctamente')
thresholds = [3, 5, 10]
aucs = pose_auc(pose_errors, thresholds)
aucs = [100.*yy for yy in aucs]
prec = 100.*np.mean(precisions)
ms = 100.*np.mean(matching_scores)
print(np.mean(matches_Correct))
print(np.mean(matches))
R=100.*(np.mean(matches_Correct)/np.mean(matches))
print('Evaluation Results (mean over {} pairs):'.format(imgCont))
print('AUC@3\t AUC@5\t AUC@10\t Rep\t ')
print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
    aucs[0], aucs[1], aucs[2], R))'''