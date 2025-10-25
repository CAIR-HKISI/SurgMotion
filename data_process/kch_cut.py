import os
from PIL import Image

# ========== 配置区域 ==========
input_root = "data/Surge_Frames/Private_KCH_Colonscopy"
output_root = "data/Surge_Frames/Private_KCH_Colonscopy_Cut"

# 定义左上角和左下角坐标
x1, y1 = 150, 0    # 左上角
x2, y2 = 100, 260  # 左下角
width = 360         # 设定要截取的宽度

# ========== 函数定义 ==========
def crop_and_save(src_path, dst_path):
    """读取、裁剪并保存图像"""
    try:
        img = Image.open(src_path)
        crop_box = (x1, y1, x1 + width, y2)
        cropped = img.crop(crop_box)
        cropped.save(dst_path)
        print(f"✅ 已保存: {dst_path}")
    except Exception as e:
        print(f"❌ 处理出错: {src_path}, 错误: {e}")

# ========== 主程序 ==========
for root, _, files in os.walk(input_root):
    for file in files:
        if file.lower().endswith(".jpg"):
            input_path = os.path.join(root, file)
            relative_path = os.path.relpath(input_path, input_root)
            
            # 目标路径保持结构
            output_path = os.path.join(output_root, relative_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            crop_and_save(input_path, output_path)

print("🎉 全部图片处理完成！")

