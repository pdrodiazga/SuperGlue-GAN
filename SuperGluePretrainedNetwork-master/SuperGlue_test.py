from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import cv2

from models.matching import Matching
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics,AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor)
torch.set_grad_enabled(False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('input', type=str, default='vid.mp4',
      help='Image directory or movie file or "camera" (for webcam).')
    parser.add_argument(
      '--output_dir', type=str, default=None,
      help='Directory where to write output frames (If None, no output)')
    parser.add_argument('--weights_path', type=str, default='superpoint_v1.pth',
        help='Path to pretrained weights file (default: superpoint_v1.pth).')
    parser.add_argument('--img_glob', type=str, default='*.png',
        help='Glob match if directory of images is specified (default: \'*.png\').')
    parser.add_argument('--skip', type=int, default=1,
        help='Images to skip if input is movie or directory (default: 1).')
    parser.add_argument('--show_extra', action='store_true',
        help='Show extra debug outputs (default: False).')
    parser.add_argument('--H', type=int, default=120,
        help='Input image height (default: 120).')
    parser.add_argument('--W', type=int, default=160,
        help='Input image width (default:160).')
    parser.add_argument('--display_scale', type=int, default=2,
        help='Factor to scale output visualization (default: 2).')
    parser.add_argument('--min_length', type=int, default=2,
        help='Minimum length of point tracks (default: 2).')
    parser.add_argument('--max_length', type=int, default=1000,
        help='Maximum length of point tracks (default: 5).')
    parser.add_argument('--nms_dist', type=int, default=4,
        help='Non Maximum Suppression (NMS) distance (default: 4).')
    parser.add_argument('--conf_thresh', type=float, default=0.015,
        help='Detector confidence threshold (default: 0.015).')
    parser.add_argument('--nn_thresh', type=float, default=0.7,
        help='Descriptor matching threshold (default: 0.7).')
    parser.add_argument('--camid', type=int, default=0,
        help='OpenCV webcam video capture ID, usually 0 or 1 (default: 0).')
    parser.add_argument('--waitkey', type=int, default=1,
        help='OpenCV waitkey time in ms (default: 1).')
    parser.add_argument('--cuda', action='store_true',
        help='Use cuda GPU to speed up network processing speed (default: False)')
    parser.add_argument('--write', action='store_true',
        help='Save output frames to a directory (default: False)')
    parser.add_argument('--write_dir', type=str, default='tracker_outputs/',
        help='Directory where to write output frames (default: tracker_outputs/).')
    parser.add_argument(
            '--superglue', choices={'indoor', 'outdoor'}, default='outdoor',
            help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=-1,
        help='Maximum number of keypoints detected by Superpoint'
            ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
        ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')
    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Show the detected keypoints')
    parser.add_argument(
        '--no_display', action='store_true',
        help='Do not display images to screen. Useful if running remotely')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')
    parser.add_argument(
            '--resize', type=int, nargs='+', default=[640, 480],
            help='Resize the input image before running inference. If two numbers, '
                'resize to the exact dimensions, if one number, resize the max '
                'dimension, if -1, do not resize')
    parser.add_argument(
        '--eval', action='store_true',
        help='Perform the evaluation'
             ' (requires ground truth pose and intrinsics)')
    
    opt = parser.parse_args()
    print(opt)
    print('==> Running SuperGlue')
    if len(opt.resize) == 2 and opt.resize[1] == -1:
            opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')
    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
            'superpoint': {
                'nms_radius': opt.nms_radius,
                'keypoint_threshold': opt.keypoint_threshold,
                'max_keypoints': opt.max_keypoints
            },
            'superglue': {
                'weights': opt.superglue,
                'sinkhorn_iterations': opt.sinkhorn_iterations,
                'match_threshold': opt.match_threshold,
            }
        }
    matching = Matching(config).eval().to(device)
    keys = ['keypoints', 'scores', 'descriptors']
    ContIteraciones=1
    for i in range(0,50):
        opt.input='.\EnchancedOri_'+str(i)+'.mp4'
        contimg=0
        vs = VideoStreamer(opt.input, opt.resize, opt.skip, opt.img_glob, opt.max_length)
        frame, ret = vs.next_frame()
        assert ret, 'Error when reading the first frame (try different --input?)'
        frame_tensor = frame2tensor(frame, device)
        last_data = matching.superpoint({'image': frame_tensor})
        last_data = {k+'0': last_data[k] for k in keys}
        last_data['image0'] = frame_tensor
        last_frame = frame
        last_image_id = 0

        if opt.output_dir is not None:
            print('==> Will write outputs to {}'.format(opt.output_dir))
            Path(opt.output_dir).mkdir(exist_ok=True)
        if opt.eval is not None:
            print('==> Will write outputs to {}'.format(opt.input))
            Path(opt.input+'_eval').mkdir(exist_ok=True)
        # Create a window to display the demo.
        if not opt.no_display:
            cv2.namedWindow('SuperGlue matches', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('SuperGlue matches', 640*2, 480)
        else:
            print('Skipping visualization, will not show a GUI.')

        print('==> Keyboard control:\n'
            '\tn: select the current frame as the anchor\n'
            '\te/r: increase/decrease the keypoint confidence threshold\n'
            '\td/f: increase/decrease the match filtering threshold\n'
            '\tk: toggle the visualization of keypoints\n'
            '\tq: quit')

        timer = AverageTimer()
        valores = [1163.45, 0., 653.626, 0., 1164.79 ,481.6, 0. ,0., 1.]

        # Crear una matriz 3x3np.array(pair[4:13]).astype(float).reshape(3, 3)
        K = np.array(valores).astype(float).reshape((3, 3))
        # Destroy all OpenCV windows
        cv2.destroyAllWindows()
        while True:
            frame, ret = vs.next_frame()
            if not ret:
                print('Finished superglue_test')
                break
            contimg+=1
            timer.update('data')
            stem0, stem1 = last_image_id, vs.i - 1
            eval_path = opt.input+"_eval/"+'{}_{}_evaluation.npz'.format(stem0, stem1)
            frame_tensor = frame2tensor(frame, device)
            pred = matching({**last_data, 'image1': frame_tensor})
            kpts0 = last_data['keypoints0'][0].cpu().numpy()
            kpts1 = pred['keypoints1'][0].cpu().numpy()
            matches = pred['matches0'][0].cpu().numpy()
            confidence = pred['matching_scores0'][0].cpu().detach().numpy()
            timer.update('forward')

            valid = matches > -1
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches[valid]]
            color = cm.jet(confidence[valid])
            text = [
                'SuperGlue',
                'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                'Matches: {}'.format(len(mkpts0))
            ]
            k_thresh = matching.superpoint.config['keypoint_threshold']
            m_thresh = matching.superglue.config['match_threshold']
            small_text = [
                'Keypoint Threshold: {:.4f}'.format(k_thresh),
                'Match Threshold: {:.2f}'.format(m_thresh),
                'Image Pair: {:06}:{:06}'.format(stem0, stem1),
            ]
            out = make_matching_plot_fast(
                last_frame, frame, kpts0, kpts1, mkpts0, mkpts1, color, text,
                path=None, show_keypoints=opt.show_keypoints, small_text=small_text)

            if not opt.no_display:
                cv2.imshow('SuperGlue matches', out)
                key = chr(cv2.waitKey(1) & 0xFF)
                if key == 'q':
                    vs.cleanup()
                    print('Exiting (via q) demo_superglue.py')
                    break
                elif key in ['e', 'r']:
                    # Increase/decrease keypoint threshold by 10% each keypress.
                    d = 0.1 * (-1 if key == 'e' else 1)
                    matching.superpoint.config['keypoint_threshold'] = min(max(
                        0.0001, matching.superpoint.config['keypoint_threshold']*(1+d)), 1)
                    print('\nChanged the keypoint threshold to {:.4f}'.format(
                        matching.superpoint.config['keypoint_threshold']))
                elif key in ['d', 'f']:
                    # Increase/decrease match threshold by 0.05 each keypress.
                    d = 0.05 * (-1 if key == 'd' else 1)
                    matching.superglue.config['match_threshold'] = min(max(
                        0.05, matching.superglue.config['match_threshold']+d), .95)
                    print('\nChanged the match threshold to {:.2f}'.format(
                        matching.superglue.config['match_threshold']))
                elif key == 'k':
                    opt.show_keypoints = not opt.show_keypoints
            last_data = {k+'0': pred[k+'1'] for k in keys}
            last_data['image0'] = frame_tensor
            last_frame = frame
            last_image_id = (vs.i - 1)
            timer.update('viz')
            timer.print()

            if opt.output_dir is not None:
                #stem = 'matches_{:06}_{:06}'.format(last_image_id, vs.i-1)
                stem = 'matches_{:06}_{:06}'.format(stem0, stem1)
                out_file = str(Path(opt.output_dir, stem + '.png'))
                print('\nWriting image to {}'.format(out_file))
                cv2.imwrite(out_file, out)

            if opt.eval:
                print(eval_path)
                try:
                    E, mask = cv2.findEssentialMat(mkpts0, mkpts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                    retval, R, t, mask = cv2.recoverPose(E, mkpts0, mkpts1, K)
                    # Matriz de transformación de la cámara relativa (extrínseca)
                    T_0to1 = np.eye(4)
                    T_0to1[:3, :3] = R
                    T_0to1[:3, 3] = t.ravel()
                    epi_errs = compute_epipolar_error(mkpts0, mkpts1, T_0to1, K, K)
                    correct = epi_errs < 5e-4
                    num_correct = np.sum(correct)
                    precision = np.mean(correct) if len(correct) > 0 else 0
                    matching_score = num_correct / len(kpts0) if len(kpts0) > 0 else 0
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
                    stem = 'matches_{:06}_{:06}'.format(stem0, stem1)
                    out_file = str(Path(opt.input+"_eval/"+ stem + '.png'))
                    print('\nWriting image to {}'.format(out_file))
                    cv2.imwrite(out_file, out)
                except:
                    print("Error en la estimación de pose.")
                    continue

        # Close any remaining windows.
        cv2.destroyAllWindows()
        if opt.eval:
            print('Proceidendo a evaluar')
            # Collate the results into a final table and print to terminal.
            pose_errors = []
            precisions = []
            matching_scores = []
            matches_Correct=[]
            N2=[]
            matches=[]
            for i in range(0,contimg-1):
                name0=str(i)
                name1=str(i+1)
                stem0, stem1 = Path(name0).stem, Path(name1).stem
                try:
                    eval_path = opt.input+"_eval/"+'{}_{}_evaluation.npz'.format(stem0, stem1)
                    results = np.load(eval_path)
                    pose_error = np.maximum(results['error_t'], results['error_R'])
                    pose_errors.append(pose_error)
                    precisions.append(results['precision'])
                    matching_scores.append(results['matching_score'])
                    matches_Correct.append(results['matches_correct'])
                    matches.append(results['matches'])
                except:
                    print("Error en la lectura de la imagen.")
                    continue
            thresholds = [3, 5, 10]
            aucs = pose_auc(pose_errors, thresholds)
            aucs = [100.*yy for yy in aucs]
            prec = 100.*np.mean(precisions)
            ms = 100.*np.mean(matching_scores)
            print(np.mean(matches_Correct))
            print(np.mean(matches))
            R=100.*(np.mean(matches_Correct)/np.mean(matches))
            print('Evaluation Results (mean over {} pairs):'.format(contimg))
            print('AUC@3\t AUC@5\t AUC@10\t Prec\t Rep\t MScore\t')
            print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
                aucs[0], aucs[1], aucs[2], prec,R, ms))
            Resultado={'AUC@3': aucs[0], 
                       'AUC@5': aucs[1], 
                       'AUC@10': aucs[2], 
                       'Prec': prec,
                       'Rep': R, 
                       'MScore': ms,
                       'error_t': results['error_t'],
                        'error_R': results['error_R'],
                        'precision': results['precision'],
                        'matching_score': results['matching_score'],
                        'num_correct': results['num_correct'],
                        'epipolar_errors': results['epipolar_errors'],
                        'matches_correct':results['matches_correct'],
                        'matches': results['matches']}
            RutaDeAUC = 'GoPro10_Eval'+"/"+'{}_Iteration.npz'.format(str(ContIteraciones))
            np.savez(str(RutaDeAUC), **Resultado)
        ContIteraciones+=1
    print('==> Finshed Demo.')