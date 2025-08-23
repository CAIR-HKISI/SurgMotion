import os
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import glob

class SurgicalMotionAnalyzer:
    def __init__(self, clip_dir, image_root_dir="/data/wjl/vjepa2/data"):
        """
        Initialize surgical motion analyzer
        
        Args:
            clip_dir: Directory containing txt files
            image_root_dir: Root directory for image files
        """
        self.clip_dir = Path(clip_dir)
        self.image_root_dir = Path(image_root_dir)
        self.current_frames = []
        self.motion_density = []
        self.current_txt_file = None
        
    def get_random_txt_file(self):
        """Randomly select a txt file"""
        txt_files = list(self.clip_dir.glob("*.txt"))
        if not txt_files:
            raise FileNotFoundError(f"No txt files found in directory {self.clip_dir}")
        
        selected_file = random.choice(txt_files)
        self.current_txt_file = selected_file
        print(f"Selected file: {selected_file}")
        return selected_file
    
    def load_frame_paths(self, txt_file):
        """Load 64 consecutive frame paths from txt file"""
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Clean paths, remove newlines and whitespace
        frame_paths = [line.strip() for line in lines if line.strip()]
        
        if len(frame_paths) < 64:
            print(f"Warning: Only {len(frame_paths)} frames found, less than 64")
        
        # Convert relative paths to absolute paths
        absolute_paths = []
        for path in frame_paths[:64]:
            if not os.path.isabs(path):
                # If relative path, build absolute path based on image_root_dir
                absolute_path = self.image_root_dir / path
            else:
                absolute_path = Path(path)
            absolute_paths.append(str(absolute_path))
        
        self.current_frames = absolute_paths
        return self.current_frames
    
    def calculate_pixel_difference(self, img1_path, img2_path):
        """Calculate pixel difference between two frames"""
        try:
            # Read images
            img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
            
            if img1 is None:
                print(f"Cannot read image1: {img1_path}")
                return 0.0
            if img2 is None:
                print(f"Cannot read image2: {img2_path}")
                return 0.0
            
            # Ensure same image dimensions
            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            
            # Calculate absolute difference
            diff = cv2.absdiff(img1, img2)
            
            # Calculate mean pixel difference
            mean_diff = np.mean(diff) / 255.0  # Normalize to 0-1
            
            return mean_diff
            
        except Exception as e:
            print(f"Error calculating pixel difference ({img1_path}, {img2_path}): {e}")
            return 0.0
    
    def generate_motion_density_curve(self):
        """Generate motion density curve for 64 frames"""
        if len(self.current_frames) < 2:
            raise ValueError("Need at least 2 frames to calculate motion density")
        
        motion_values = []
        valid_calculations = 0
        
        print("Starting motion density calculation...")
        for i in range(len(self.current_frames) - 1):
            # Calculate pixel difference between adjacent frames
            diff = self.calculate_pixel_difference(
                self.current_frames[i], 
                self.current_frames[i + 1]
            )
            motion_values.append(diff)
            if diff > 0:
                valid_calculations += 1
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(self.current_frames)-1} frame pairs")
        
        # Add value for last frame (use second-to-last value)
        if motion_values:
            motion_values.append(motion_values[-1])
        
        print(f"Calculation complete, valid calculations: {valid_calculations}/{len(motion_values)-1}")
        self.motion_density = motion_values
        return motion_values
    
    def load_actual_images(self, frame_indices):
        """Load actual surgical images"""
        images = []
        for idx in frame_indices:
            if idx < len(self.current_frames):
                img_path = self.current_frames[idx]
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        # Resize image for display
                        img = cv2.resize(img, (320, 240))
                        images.append(img)
                    else:
                        # If unable to load, create placeholder image
                        placeholder = self.create_placeholder_image(f"Frame {idx+1}\nNot Found")
                        images.append(placeholder)
                except Exception as e:
                    print(f"Failed to load image {img_path}: {e}")
                    placeholder = self.create_placeholder_image(f"Frame {idx+1}\nError")
                    images.append(placeholder)
            else:
                placeholder = self.create_placeholder_image(f"Frame {idx+1}\nOut of Range")
                images.append(placeholder)
        
        return images
    
    def create_placeholder_image(self, text):
        """Create placeholder image"""
        img = np.ones((240, 320, 3), dtype=np.uint8) * 128
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_lines = text.split('\n')
        y_start = 100
        for i, line in enumerate(text_lines):
            cv2.putText(img, line, (50, y_start + i*30), font, 0.6, (255, 255, 255), 2)
        return img
    
    def plot_analysis_grid(self, output_path="surgical_motion_analysis.png"):
        """
        Plot alternating surgical images and motion density curves
        Odd rows: 8 surgical images
        Even rows: corresponding motion density curves
        """
        if not self.motion_density:
            print("Please generate motion density curve first")
            return
        
        # Select frame indices to display (evenly distributed across 64 frames)
        frame_indices = [i * 8 for i in range(8)]  # 0, 8, 16, 24, 32, 40, 48, 56
        
        # Load actual images
        actual_images = self.load_actual_images(frame_indices)
        
        # Create figure
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle('Surgical Video Motion Density Analysis\n64 Consecutive Frames Pixel Difference and Motion Density Visualization', 
                     fontsize=16, fontweight='bold')
        
        # Add file information
        if self.current_txt_file:
            fig.text(0.5, 0.94, f'Analysis File: {self.current_txt_file.name}', 
                    ha='center', fontsize=10, style='italic')
        
        # Layout: 2 rows, each row has 8 columns
        rows = 2
        cols = 8
        
        # First row: surgical images
        for col in range(cols):
            frame_idx = frame_indices[col]
            ax = plt.subplot(rows, cols, col + 1)
            
            if col < len(actual_images):
                # Display actual surgical image
                ax.imshow(actual_images[col])
                ax.set_title(f'Frame {frame_idx + 1}', fontsize=11, fontweight='bold')
                ax.axis('off')
                
                # Add motion intensity annotation
                if frame_idx < len(self.motion_density):
                    intensity = self.motion_density[frame_idx]
                    color = 'red' if intensity > 0.1 else 'green'
                    ax.text(0.02, 0.02, f'{intensity:.3f}', 
                           transform=ax.transAxes, fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                           verticalalignment='bottom', color='white', fontweight='bold')
        
        # Second row: motion density curves
        for col in range(cols):
            ax = plt.subplot(rows, cols, cols + col + 1)
            
            # Calculate current segment data range (8 frames per segment)
            start_frame = col * 8
            end_frame = min(start_frame + 8, len(self.motion_density))
            
            if start_frame < len(self.motion_density):
                # Plot motion density curve segment
                x_data = list(range(start_frame, end_frame))
                y_data = self.motion_density[start_frame:end_frame]
                
                # Plot curve
                ax.plot(x_data, y_data, 'b-', linewidth=2.5, marker='o', markersize=4, alpha=0.8)
                ax.fill_between(x_data, y_data, alpha=0.2, color='blue')
                
                # Highlight peaks
                if y_data:
                    max_idx = np.argmax(y_data)
                    ax.plot(x_data[max_idx], y_data[max_idx], 'ro', markersize=6, alpha=0.8)
                
                # Set axes
                ax.set_xlim(start_frame, start_frame + 7)
                y_max = max(0.5, max(self.motion_density)) * 1.1
                ax.set_ylim(0, y_max)
                ax.set_title(f'Motion Density\nFrames {start_frame + 1}-{end_frame}', 
                           fontsize=10, fontweight='bold')
                ax.grid(True, alpha=0.4, linestyle='--')
                ax.set_xlabel('Frame Index', fontsize=9)
                ax.set_ylabel('Motion\nIntensity', fontsize=9)
                
                # Add statistics
                if y_data:
                    avg_motion = np.mean(y_data)
                    max_motion = np.max(y_data)
                    ax.text(0.02, 0.98, f'Avg: {avg_motion:.3f}\nMax: {max_motion:.3f}', 
                           transform=ax.transAxes, fontsize=8,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9),
                           verticalalignment='top')
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.88, hspace=0.3, wspace=0.25)
        
        # Save image
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Analysis result saved to: {output_path}")
        plt.show()
    
    def print_statistics(self):
        """Print motion density statistics"""
        if not self.motion_density:
            print("No motion density data generated")
            return
        
        motion_array = np.array(self.motion_density)
        stats = {
            'Total Frames': len(self.motion_density),
            'Average Motion Intensity': np.mean(motion_array),
            'Maximum Motion Intensity': np.max(motion_array),
            'Minimum Motion Intensity': np.min(motion_array),
            'Standard Deviation': np.std(motion_array),
            'High Motion Frames (>0.1)': np.sum(motion_array > 0.1),
            'Medium Motion Frames (0.05-0.1)': np.sum((motion_array >= 0.05) & (motion_array <= 0.1)),
            'Low Motion Frames (<0.05)': np.sum(motion_array < 0.05)
        }
        
        print("\n" + "="*60)
        print("Motion Density Statistics")
        print("="*60)
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key:25s}: {value:.6f}")
            else:
                print(f"{key:25s}: {value}")
        print("="*60)

def main():
    """Main function"""
    # Set parameters
    clip_dir = "/data/wjl/vjepa2/data/Surge_Frames/AutoLaparo/clips_64f/clip_dense_64f_info/train"
    image_root_dir = "/data/wjl/vjepa2"
    
    try:
        # Create analyzer
        analyzer = SurgicalMotionAnalyzer(clip_dir, image_root_dir)
        
        # Randomly select txt file
        txt_file = analyzer.get_random_txt_file()
        
        # Load frame paths
        frame_paths = analyzer.load_frame_paths(txt_file)
        print(f"Loaded {len(frame_paths)} frame paths")
        
        # Check image file accessibility
        print("\nChecking image file accessibility:")
        for i, path in enumerate(frame_paths[:5]):
            exists = os.path.exists(path)
            print(f"Frame {i+1}: {exists} - {path}")
        
        # Generate motion density curve
        motion_curve = analyzer.generate_motion_density_curve()
        print(f"Generated {len(motion_curve)} motion density values")
        
        # Print statistics
        analyzer.print_statistics()
        
        # Plot analysis grid
        output_filename = f"surgical_motion_analysis_{txt_file.stem}.png"
        analyzer.plot_analysis_grid(output_filename)
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Set font for better display
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    main()

