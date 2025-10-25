import os
import pandas as pd
from tqdm import tqdm

# ================= 路径配置 =================
FRAME_PATH = "data/Surge_Frames/OphNet2024_phase/frames"
OUT_DIR = "data/Surge_Frames/OphNet2024_phase"
os.makedirs(OUT_DIR, exist_ok=True)

LABEL_PATH = "data/Ophthalmology/OphNet2024_trimmed_phase/OphNet2024_loca_challenge_phase.csv"

# ================= Phase ID → 英文名称映射 =================
phase2name = {
    0: "Viscoelastic Injection",
    1: "Nuclear Management (for cataract surgery)",
    2: "Step Interval",
    3: "Non-functional Segment",
    4: "Intraocular Lens Implantation",
    5: "Incision Closure",
    6: "Corneal Incision Creation",
    7: "Capsulorhexis",
    8: "Cortex Aspiration",
    9: "Recipient Corneal Bed Preparation",
    10: "Anterior Chamber Injection/Washing",
    11: "Viscoelastic Aspiration",
    12: "Capsular Membrane Staining",
    13: "Corneal Graft Suturing",
    14: "Viscoelastic Application on Cornea",
    15: "Corneal-Scleral Tunnel Creation",
    16: "Swab Wiping",
    17: "Conjunctival Incision Creation",
    18: "Anterior Chamber Gas Injection",
    19: "Ocular Surface Irrigation",
    20: "Surgical Marking",
    21: "Subconjunctival Drug Injection",
    22: "Donor Corneal Graft Preparation",
    23: "Scleral Hemostasis",
    24: "Corneal Interlamellar Injection",
    25: "Drainage Tube Implantation",
    26: "Use of Iris Expander",
    27: "Drainage Device Preparation",
    28: "Scleral Flap Creation",
    29: "Anterior Chamber Washing",
    30: "Suspension Suture",
    31: "Drainage Device Implantation",
    32: "Intraoperative Gonioscopy Application",
    33: "Corneal Measurement",
    34: "Scleral Flap Suturing",
    35: "Scleral Tunnel Creation",
    36: "Placement of Bandage Contact Lens",
    37: "Peripheral Iridectomy",
    38: "Anterior Chamber Drainage Device Implantation",
    39: "Antimetabolite Application",
    40: "Scleral Support Ring Manipulation",
    41: "Deep Sclerectomy",
    42: "Anterior Vitrectomy",
    43: "Placement of Eyelid Speculum",
    44: "Hooking of Extraocular Muscle",
    45: "Observation of Corneal Astigmatism",
    46: "Pupil Dilation",
    47: "Iris Prolapse Management",
    48: "Measurement on the Scleral",
    49: "Goniotomy",
    50: "Trabeculectomy",
    51: "Instrument Fabrication",
    52: "Capsular Tension Ring Implantation",
    53: "Femtosecond Laser-Assisted Corneal Transplantation",
    54: "Removal of Pupillary/Iris Fibrosis Membrane",
    55: "Allograft/Biological Tissue Trimming",
    56: "Corneal Suture Removal",
    57: "Iris Synechiae Separation",
    58: "Placement of Sponge on Cornea",
    59: "Femtosecond Laser-Assisted Cataract Surgery",
    60: "Astigmatism Axis Gauge",
    61: "Microcatheter Insertion into Trabecular Meshwork",
    62: "Sub-Iris Exploration",
    63: "Artificial Cornea Manipulaiton",
    64: "Scleral Puncture/Incision",
    65: "Astigmatic Keratotomy",
    66: "Scleral Fixation of Intraocular Lens",
    67: "Removal of Fascia Tissue",
    68: "Canthotomy",
    69: "Removal of Lens Fibrotic Membrane",
    70: "Amniotic Membrane Transplantation",
    71: "Pupilloplasty",
    72: "Anterior Chamber Irrigation",
    73: "Sub-Tenon Injection",
    74: "Sclerectomy",
    75: "Corneal Interlamellar Irrigation",
    76: "Cyclophotocoagulation",
    77: "Drainage Device Adjustment",
    78: "Schlemm's Canal Inner Wall Removal",
    79: "Scleral Flap Incision Inspection",
    80: "Lens Extraction",
    81: "Anterior Chamber Vitreous Cleaning",
    82: "Intraocular Lens Removal",
    83: "Pterygium Excision",
    84: "Scleral Suture",
    85: "Drainage Tube Removal",
    86: "Suture Trimming",
    87: "Conjunctival Vessel Examination",
    88: "Special Puncture Knife Traversing the Anterior Chamber",
    89: "Removal of Object from Anterior Chamber",
    90: "Anterior Chamber Inspection",
    91: "Iris Repositioning",
    92: "Suprachoroidal Space Separation",
    93: "Scleral Flap Embedding in the Suprachoroidal Space",
    94: "Vitreoretinal Surgery",
    95: "Conjunctival Trimming",
}


def generate_ophnet_csv():
    if not os.path.exists(LABEL_PATH):
        print(f"❌ Label file not found: {LABEL_PATH}")
        return

    df_labels = pd.read_csv(LABEL_PATH)
    df_labels = df_labels.sort_values(["video_id", "start"]).reset_index(drop=True)
    df_labels["clip_index"] = df_labels.groupby("video_id").cumcount()

    all_data = []
    missing_frames = []
    global_idx = 0

    print(f"📊 Found {len(df_labels)} clips across {df_labels['video_id'].nunique()} videos")

    # === 遍历每个 OphNet clip ===
    for _, row in tqdm(df_labels.iterrows(), total=len(df_labels), desc="Processing OphNet clips"):
        video_id = row["video_id"]
        clip_index = row["clip_index"]
        split = row["split"]
        phase_id = int(row["phase_id"])
        phase_name = phase2name.get(phase_id, "Unknown Phase")

        # === 帧路径结构 ===
        case_name = f"{video_id}_{clip_index}"
        video_dir = os.path.join(FRAME_PATH, case_name)
        if not os.path.exists(video_dir):
            print(f"⚠️ Missing frame directory: {video_dir}")
            continue

        frame_files = sorted([f for f in os.listdir(video_dir) if f.endswith(".jpg")])
        if not frame_files:
            print(f"⚠️ No frames found in: {video_dir}")
            continue

        # === 遍历帧文件 ===
        for frame_file in frame_files:
            frame_path = os.path.join(video_dir, frame_file)
            if not os.path.exists(frame_path):
                missing_frames.append({
                    "Split": split,
                    "Video_ID": video_id,
                    "Case_Name": case_name,
                    "Missing_Frame": frame_file
                })
                continue

            try:
                frame_id = int(frame_file.split("_")[-1].split(".")[0])
            except Exception:
                frame_id = None

            data_item = {
                "index": global_idx,
                "DataName": "OphNet2024_phase",
                "Year": 2024,
                "Case_Name": case_name,
                "Case_ID": int(video_id.replace("case_", "")),
                "Frame_Path": frame_path,
                "Phase_GT": phase_id,
                "Phase_Name": phase_name,
                "Split": split
            }
            all_data.append(data_item)
            global_idx += 1

    # === 汇总输出 ===
    if not all_data:
        print("❌ No valid data found.")
        return

    df = pd.DataFrame(all_data)
    print(f"\n✅ Total processed frames: {len(df)}")

    merged_csv = os.path.join(OUT_DIR, "metadata.csv")
    df.to_csv(merged_csv, index=False)
    print(f"💾 Saved {merged_csv}")

    # === 分 Split 保存 ===
    for split in df["Split"].unique():
        df_split = df[df["Split"] == split]
        split_csv = os.path.join(OUT_DIR, f"{split}_metadata.csv")
        df_split.to_csv(split_csv, index=False)
        print(f"💾 Saved {len(df_split)} frames to {split_csv}")

    # === 缺帧报告 ===
    if missing_frames:
        df_miss = pd.DataFrame(missing_frames)
        miss_csv = os.path.join(OUT_DIR, "missing_frames_report.csv")
        df_miss.to_csv(miss_csv, index=False)
        print(f"\n❌ Missing frames detected! Saved report to: {miss_csv}")
        print(df_miss.groupby("Case_Name").size())
    else:
        print("\n✅ All frames verified exist!")

    print("\n✅ OphNet metadata generation completed successfully!")


if __name__ == "__main__":
    generate_ophnet_csv()

