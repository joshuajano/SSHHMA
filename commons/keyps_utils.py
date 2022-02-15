import numpy as np
def j3d_processing(S, r, f):
    """Process gt 3D keypoints and apply all augmentation transforms."""
    rot_mat = np.eye(3)
    if not r == 0:
        rot_rad = -r * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
    S[:, :-1] = np.einsum('ij,kj->ki', rot_mat, S[:, :-1]) 
    if f:
        S = flip_kp(S)
    S = S.astype('float32')
    return S
def j2d_processing(kp, center, scale, r, f, IMG_RES=256):
    nparts = kp.shape[0]
    for i in range(nparts):
        kp[i,0:2] = transform(kp[i,0:2]+1, center, scale, 
                              [IMG_RES, IMG_RES], rot=r)
    kp[:,:-1] = 2.*kp[:,:-1]/IMG_RES - 1.
    if f:
        kp = flip_kp(kp)
    kp = kp.astype('float32')
    return kp
def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t
def transform(pt, center, scale, res, invert=0, rot=0):
    """Transform pixel location to different reference."""
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0]-1, pt[1]-1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int)+1

def mapping_keypoints(keyps, indexes, 
            body_idxs, lhand_idxs, rhand_idxs, face_idxs,
            binarization = True, use_face_contour=False, 
            body_thresh = 0.1, face_thresh = 0.4, hand_thresh = 0.2):
    if keyps.shape[1]>3:
        output_keyps = np.zeros([127 + 17 * use_face_contour,  4], dtype=np.float32)
    else:
        output_keyps = np.zeros([127 + 17 * use_face_contour,  3], dtype=np.float32)
    # output_keyps = np.zeros([127 + 17 * use_face_contour,  3], dtype=np.float32)
    output_keyps[indexes['tgt_idxs']] = keyps[indexes['src_idxs']]
    output_keyps[output_keyps[:, -1] < 0, -1] = 0
    body_conf = output_keyps[body_idxs, -1]
    left_hand_conf = output_keyps[lhand_idxs, -1]
    right_hand_conf = output_keyps[rhand_idxs, -1]
    # face_conf = output_keyps[face_idxs, -1]
    body_conf[body_conf < body_thresh] = 0.0
    body_conf[body_conf > body_thresh] = 1.0
    left_hand_conf[left_hand_conf < hand_thresh] = 0.0
    right_hand_conf[right_hand_conf < hand_thresh] = 0.0
    # face_conf[face_conf < face_thresh] = 0.0

    if binarization:
        body_conf = (
            body_conf >= body_thresh).astype(
                output_keyps.dtype)
        left_hand_conf = (
            left_hand_conf >= hand_thresh).astype(
                output_keyps.dtype)
        right_hand_conf = (
            right_hand_conf >= hand_thresh).astype(
                output_keyps.dtype)
        # face_conf = (
        #     face_conf >= face_thresh).astype(
        #         output_keyps.dtype)
    output_keyps[body_idxs, -1] = body_conf
    output_keyps[lhand_idxs, -1] = left_hand_conf
    output_keyps[rhand_idxs, -1] = right_hand_conf
    # output_keyps[face_idxs, -1] = face_conf
    return output_keyps
def mapping_batch_keypoints(keyps, indexes, seqlen,  
                    face_contour = True, binarization = True, 
                    body_thresh = 0.1, face_thresh = 0.4, hand_thresh = 0.2):
    output_keyps = np.zeros([seqlen, 127 + 17 * face_contour,  3], dtype=np.float32)
    output_keyps[:, indexes['tgt_idxs']] = keyps[:, indexes['src_idxs']]

    # output_keyps[:, output_keyps[:, :, -1] < 0, -1] = 0
    
    # body_conf = output_keyps[:, indexes['body_idxs'], -1]
    # left_hand_conf = output_keyps[:, indexes['left_hand_idxs'], -1]
    # right_hand_conf = output_keyps[:, indexes['right_hand_idxs'], -1]
    # face_conf = output_keyps[:, indexes['face_idxs'], -1]

    # output_keyps[:, indexes['body_idxs'], -1] = body_conf
    # output_keyps[:, indexes['left_hand_idxs'], -1] = left_hand_conf
    # output_keyps[:, indexes['right_hand_idxs'], -1] = right_hand_conf
    # output_keyps[:, indexes['face_idxs'], -1] = face_conf
    return output_keyps

    # body_conf[body_conf < body_thresh] = 0.0
    # body_conf[body_conf > body_thresh] = 1.0
    # left_hand_conf[left_hand_conf < hand_thresh] = 0.0
    # right_hand_conf[right_hand_conf < hand_thresh] = 0.0
    # face_conf[face_conf < face_thresh] = 0.0

    # if binarization:
    #     body_conf = (
    #         body_conf >= body_thresh).astype(
    #             output_keyps.dtype)
    #     left_hand_conf = (
    #         left_hand_conf >= hand_thresh).astype(
    #             output_keyps.dtype)
    #     right_hand_conf = (
    #         right_hand_conf >= hand_thresh).astype(
    #             output_keyps.dtype)
    #     face_conf = (
    #         face_conf >= face_thresh).astype(
    #             output_keyps.dtype)
    
    

def mapping_keyps(keyps, indexes, 
                    face_contour = True, binarization = True, 
                    body_thresh = 0.1, face_thresh = 0.4, hand_thresh = 0.2):
    if keyps.shape[1]==3:
        output_keyps = np.zeros([127 + 17 * face_contour,  3], dtype=np.float32)
    elif keyps.shape[1]==4:
        output_keyps = np.zeros([127 + 17 * face_contour,  4], dtype=np.float32)
    output_keyps[indexes['tgt_idxs']] = keyps[indexes['src_idxs']]
    output_keyps[output_keyps[:, -1] < 0, -1] = 0
    body_conf = output_keyps[indexes['body_idxs'], -1]
        
    left_hand_conf = output_keyps[indexes['left_hand_idxs'], -1]
    right_hand_conf = output_keyps[indexes['right_hand_idxs'], -1]
    
    face_conf = output_keyps[indexes['face_idxs'], -1]
        
    body_conf[body_conf < body_thresh] = 0.0
    body_conf[body_conf > body_thresh] = 1.0
    left_hand_conf[left_hand_conf < hand_thresh] = 0.0
    right_hand_conf[right_hand_conf < hand_thresh] = 0.0
    face_conf[face_conf < face_thresh] = 0.0
    if binarization:
        body_conf = (
            body_conf >= body_thresh).astype(
                output_keyps.dtype)
        left_hand_conf = (
            left_hand_conf >= hand_thresh).astype(
                output_keyps.dtype)
        right_hand_conf = (
            right_hand_conf >= hand_thresh).astype(
                output_keyps.dtype)
        face_conf = (
            face_conf >= face_thresh).astype(
                output_keyps.dtype)
    output_keyps[indexes['body_idxs'], -1] = body_conf
    output_keyps[indexes['left_hand_idxs'], -1] = left_hand_conf
    output_keyps[indexes['right_hand_idxs'], -1] = right_hand_conf
    output_keyps[indexes['face_idxs'], -1] = face_conf
    return output_keyps

