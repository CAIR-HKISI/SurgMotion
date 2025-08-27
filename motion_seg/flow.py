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

def flow_to_color(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[..., 0], flow[..., 1]
    mag, ang = cv2.cartToPolar(fx, fy)
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 2] = 255
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return rgb

def draw_motion_vector_on_img(img, motion_vec, color=(0,0,255), thickness=4):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    scale = 10
    end = (int(center[0] + motion_vec[0]*scale), int(center[1] + motion_vec[1]*scale))
    vis = img.copy()
    cv2.arrowedLine(vis, center, end, color, thickness, tipLength=0.2)
    cv2.putText(vis, f"Motion: ({motion_vec[0]:.2f},{motion_vec[1]:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return vis

def safe_imread(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {path}")
    return img

def save_dense_flow_with_motion_vec(frame_paths, out_dir, use_phasecorr=False):
    os.makedirs(out_dir, exist_ok=True)
    imgs = [safe_imread(p) for p in frame_paths]
    prev_gray = cv2.cvtColor(imgs[0], cv2.COLOR_BGR2GRAY)
    h, w = prev_gray.shape

    for i in range(1, len(imgs)):
        curr_gray = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY)
        # --- 稠密光流
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray,
                                            None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # --- 方式1：光流平均/方式2：相位相关
        if use_phasecorr:
            shift, _ = cv2.phaseCorrelate(np.float32(prev_gray), np.float32(curr_gray))
            motion_vec = shift
        else:
            dx = np.mean(flow[...,0])
            dy = np.mean(flow[...,1])
            motion_vec = (dx, dy)

        # --- 彩色光流可视化
        flow_vis = flow_to_color(flow)
        # 使用原始帧名更易追溯
        base_name = os.path.splitext(os.path.basename(frame_paths[i]))[0]
        out_flow_vis = os.path.join(out_dir, f"{base_name}_flow_color.jpg")
        cv2.imwrite(out_flow_vis, flow_vis)

        # --- 运动矢量箭头可视化
        motion_vis = draw_motion_vector_on_img(imgs[i], motion_vec)
        out_arrow_vis = os.path.join(out_dir, f"{base_name}_motion_vec.jpg")
        cv2.imwrite(out_arrow_vis, motion_vis)

        print(f"Saved: {out_flow_vis}, {out_arrow_vis}")
        prev_gray = curr_gray

if __name__ == "__main__":
    # 配置路径
    clip_dir = "data/Surge_Frames/Cholec80/clips_8f/clip_dense_8f_info/train"
    out_flow_dir = "logs/motion_seg/flow_vis"
    clip_txts = sorted(os.listdir(clip_dir))
    clip_txt_path = os.path.join(clip_dir, clip_txts[100])
    frame_paths = load_clip_frames(clip_txt_path)
    # use_phasecorr=True 使用相位相关法估计全局运动，否则用平均光流
    save_dense_flow_with_motion_vec(frame_paths, out_flow_dir, use_phasecorr=False)