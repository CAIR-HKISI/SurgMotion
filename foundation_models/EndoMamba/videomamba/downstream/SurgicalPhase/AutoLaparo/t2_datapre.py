import os
import pickle
import cv2


def get_3fps_frame(video_src_path, frame_save_path):
	video_path = sorted(os.listdir(video_src_path))
	for videos in video_path:

		video_path = video_src_path + videos
		video_name = videos[:3]
		if not os.path.exists(frame_save_path + video_name):
			os.mkdir(frame_save_path + video_name)

		cap = cv2.VideoCapture(video_path)
		frame_num = 0
		while cap.isOpened():
			ret, frame = cap.read()
			if not ret:
				break

			frame_num += 1
			if frame_num in [1,9,17,25,33,41,50,58,66,75,83,91,100,108,116,125]:  # down_sample from 25 to 3 fps
				img_save_path = frame_save_path + video_name + '/' + str(frame_num).zfill(3) + ".jpg"
				img_result = cv2.resize(frame, (250, 250))
				cv2.imwrite(img_save_path, img_result)
				cv2.waitKey(1)

		cap.release()
		print("Video {:s}: Totally have {:d} frames".format(video_name, frame_num))


def get_img_label(label_src_dir, frame_save_path, data_save_path):
	train_val_test_all = []
	motion_label_info_all = []
	img_all = []

	with open(label_src_dir, 'r') as f:
		lines = f.readlines()[1:]  # skip the first line

		for line in lines:
			motion_label_info = int(line.split('\t')[1])
			motion_label_info_all.append([motion_label_info])

	clip_ls = sorted(os.listdir(frame_save_path))
	for clip in clip_ls:
		clip_path = os.path.join(frame_save_path, clip)
		clip_img_ls = sorted(os.listdir(clip_path))
		clip_img_list = []
		for clip_img in clip_img_ls:
			clip_img_path = os.path.join(clip_path, clip_img)
			clip_img_list.append(clip_img_path)
		img_all.append(clip_img_list)

	print(len(motion_label_info_all), len(img_all))

	train_img_path = [img_all[i] for i in range(170)]
	val_img_path = [img_all[i] for i in range(170, 227)]
	test_img_path = [img_all[i] for i in range(227, 300)]

	train_label = [motion_label_info_all[i] for i in range(170)]
	val_label = [motion_label_info_all[i] for i in range(170, 227)]
	test_label = [motion_label_info_all[i] for i in range(227, 300)]

	train_val_test_all.append([train_img_path, val_img_path, test_img_path])
	train_val_test_all.append([train_label, val_label, test_label])
	with open(data_save_path, 'wb') as f:
		pickle.dump(train_val_test_all, f)
	print("Saved")


if __name__ == "__main__":
	video_src_path = "./video_src_path/"
	label_src_dir = "./laparoscope_motion_label.txt"
	frame_save_path = "./frame_save_path/"
	data_save_path = "./data_save_path/laparoscope_motion_data_train170_val57_test73.pkl"
	if not os.path.exists(frame_save_path):
		os.mkdir(frame_save_path)

	get_3fps_frame(video_src_path, frame_save_path)
	get_img_label(label_src_dir, frame_save_path, data_save_path)
