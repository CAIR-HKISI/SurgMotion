import os
import cv2
import numpy as np

def load_clip_frames(clip_txt_path):
    frame_paths = []
    with open(clip_txt_path, "r") as f:
        for line in f:
            frame_path = line.strip().split(" ")[0]
            frame_paths.append(frame_path)
    return frame_paths

def orb_track_with_matches(frame_paths, max_points=300):
    """
    用ORB对相邻帧做特征点检测与匹配，构建轨迹
    返回：每帧图片、每帧跟踪点（按轨迹id排列）
    """
    orb = cv2.ORB_create(nfeatures=1000)
    # 注意：ORB特征是uint8类型，需要用 LSH 匹配
    index_params = dict(algorithm=6,  # FLANN_INDEX_LSH
                        table_number=6,
                        key_size=12,
                        multi_probe_level=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    imgs = [cv2.imread(p) for p in frame_paths]
    kps_list, descs_list = [], []
    for img in imgs:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps, descs = orb.detectAndCompute(gray, None)
        kps_list.append(kps)
        descs_list.append(descs)
    if len(kps_list[0]) == 0:
        raise RuntimeError("No keypoints detected in the first frame.")

    kp0 = sorted(kps_list[0], key=lambda x: -x.response)[:max_points]
    desc0 = descs_list[0][:len(kp0)]
    points_tracks = [[kp.pt] for kp in kp0]

    prev_kps = kp0
    prev_descs = desc0

    for i in range(1, len(imgs)):
        curr_kps = kps_list[i]
        curr_descs = descs_list[i]
        if curr_descs is None or len(curr_descs) == 0:
            for track in points_tracks:
                track.append(None)
            continue
        # dtype必须uint8
        matches = flann.knnMatch(np.asarray(prev_descs, np.uint8), np.asarray(curr_descs, np.uint8), k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.85 * n.distance:  # ORB推荐略宽松
                good.append(m)
        print(f"Frame {i}: {len(good)} good matches")
        curr_pts_matched = [None] * len(points_tracks)
        idx_map = {}
        for g in good:
            track_idx = g.queryIdx
            kp = curr_kps[g.trainIdx]
            curr_pts_matched[track_idx] = kp.pt
            idx_map[track_idx] = g.trainIdx
        for t in range(len(points_tracks)):
            points_tracks[t].append(curr_pts_matched[t])
        matched_indices = [idx_map[t] for t in range(len(points_tracks)) if t in idx_map]
        prev_kps = [curr_kps[j] for j in matched_indices]
        prev_descs = np.asarray([curr_descs[j] for j in matched_indices], np.uint8)
    return imgs, points_tracks

def make_big_track_image(imgs, points_tracks, out_path):
    n = len(imgs)
    h, w = imgs[0].shape[:2]
    big_img = np.ones((h, w * n, 3), dtype=np.uint8) * 255
    for i, img in enumerate(imgs):
        big_img[:, i*w:(i+1)*w, :] = img
    colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(len(points_tracks))]
    for pt_idx, track in enumerate(points_tracks):
        color = colors[pt_idx]
        prev = None
        for i, pt in enumerate(track):
            if pt is not None:
                cx, cy = int(pt[0]) + i * w, int(pt[1])
                cv2.circle(big_img, (cx, cy), 2, color, -1)
                if prev is not None:
                    cv2.line(big_img, prev, (cx, cy), color, 1)
                prev = (cx, cy)
            else:
                prev = None
    cv2.imwrite(out_path, big_img)
    print(f"大图已保存：{out_path}")

if __name__ == "__main__":
    clip_dir = "data/Surge_Frames/Cholec80/clips_8f/clip_dense_8f_info/train"
    out_img_path = "logs/motion_seg/big_track_orb_demo.jpg"
    clip_txts = sorted(os.listdir(clip_dir))
    clip_txt_path = os.path.join(clip_dir, clip_txts[100])
    frame_paths = load_clip_frames(clip_txt_path)
    imgs, points_tracks = orb_track_with_matches(frame_paths, max_points=300)
    make_big_track_image(imgs, points_tracks, out_img_path)

