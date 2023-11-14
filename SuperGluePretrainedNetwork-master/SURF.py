import cv2
import numpy as np
from pathlib import Path
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics,AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor)

# Reemplaza 'tu_video.mp4' con la ruta o el nombre de tu video
video_path = 'EnchancedOri1.mp4'

# Inicializar el detector SURF
surf = cv2.SIFT_create()

# Inicializar el matcher FLANN (Fast Library for Approximate Nearest Neighbors)
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Capturar el video
cap = cv2.VideoCapture(video_path)

# Capturar el primer fotograma para usar como referencia
ret, first_frame = cap.read()
first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
keypoints1, descriptors1 = surf.detectAndCompute(first_gray, None)
valores = [1163.45, 0., 653.626, 0., 1164.79 ,481.6, 0. ,0., 1.]
K = np.array(valores).astype(float).reshape((3, 3))
imgCont=0
while True:
    # Capturar el fotograma actual
    ret, frame = cap.read()

    # Salir del bucle si no hay más fotogramas
    if not ret:
        break
    stem0, stem1 = imgCont,imgCont+1
    eval_path = "Surf_Eval/"+'{}_{}_evaluation.npz'.format(stem0, stem1)
    imgCont+=1
    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    keypoints1, descriptors1 = surf.detectAndCompute(first_gray, None)  
    # Encontrar los puntos clave y descripciones en el fotograma actual
    keypoints2, descriptors2 = surf.detectAndCompute(gray, None)

    # Aplicar el matcher FLANN para encontrar emparejamientos
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Aplicar el test de ratio para mantener solo buenos emparejamientos
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    mkpts0 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    mkpts1 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    mkpts0 = np.array(mkpts0)[:, 0, :]
    mkpts1 =np.array(mkpts1)[:, 0, :]
    # Dibujar los emparejamientos en el fotograma actual
    frame_with_matches = cv2.drawMatches(first_frame, keypoints1, frame, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Mostrar el fotograma con emparejamientos
    cv2.imshow('Video con Emparejamientos SURF (FLANN)', frame_with_matches)
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
    matching_score = num_correct / len(keypoints2) if len(keypoints2) > 0 else 0
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
    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
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
        eval_path = "Surf_Eval/"+'{}_{}_evaluation.npz'.format(stem0, stem1)
        results = np.load(eval_path)
        pose_error = np.maximum(results['error_t'], results['error_R'])
        pose_errors.append(pose_error)
        precisions.append(results['precision'])
        matching_scores.append(results['matching_score'])
        matches_Correct.append(results['matches_correct'])
        matches.append(results['matches'])
    except:
        print("La imagen "+"Surf_Eval/"+'{}_{}_evaluation.npz'.format(stem0, stem1)+' no se guardo correctamente')

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