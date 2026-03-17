import cv2
import numpy as np
import os
import turret_submission as student

def run_autograder():
    # --- CONFIGURATION ---
    gt_file = "assignment_1/assets/turret_centers_gt.txt"
    frames_dir = "assignment_1/assets/turret_frames_gt"
    pixel_tolerance = 50.0  # Allowable Euclidean distance error in pixels
    success_threshold = 0.95  # 95% of frames must pass to be considered correct

    print("--- Starting Visual Servoing Autograder ---")
    
    # Check for required files
    if not os.path.exists(gt_file):
        print(f"❌ Error: Ground truth file '{gt_file}' not found.")
        return

    if not os.path.isdir(frames_dir):
        print(f"❌ Error: Frames directory '{frames_dir}' not found at {os.path.abspath(frames_dir)}")
        return

    # Load Ground Truth Data
    with open(gt_file, "r") as f:
        gt_lines = [line.strip() for line in f.readlines() if line.strip()]

    total_test_frames = len(gt_lines)
    passed_frames = 0
    fail_details = []

    if total_test_frames == 0:
        print("❌ Error: Ground truth file is empty.")
        return

    print(f"Loaded {total_test_frames} ground truth entries. Evaluating implementation...")

    for line in gt_lines:
        try:
            parts = line.split(',')
            if len(parts) != 3: continue
            
            frame_idx, tx, ty = map(int, parts)
            
            frame_path = os.path.join(frames_dir, f"frame_{frame_idx:04d}.png")
            
            if not os.path.exists(frame_path):
                continue

            bgr_img = cv2.imread(frame_path)            
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

            # Call student implementation
            cx, cy = student.find_target(rgb_img)

            if cx is None or cy is None:
                if len(fail_details) < 5:
                    fail_details.append(f"Frame {frame_idx}: Target not detected.")
                continue

            # Calculate Euclidean Distance Error
            dist = np.sqrt((cx - tx)**2 + (cy - ty)**2)

            if dist <= pixel_tolerance:
                passed_frames += 1
            else:
                if len(fail_details) < 5:
                    fail_details.append(f"Frame {frame_idx}: Low accuracy. Error: {dist:.2f}px")

        except Exception as e:
            print(f"⚠️ Error processing line '{line}': {e}")

    # --- FINAL REPORTING ---
    success_rate = (passed_frames / total_test_frames) if total_test_frames > 0 else 0
    
    print("\n" + "="*57)
    print(f"{'AUTOGRADER RESULTS':^57}")
    print("="*57)
    print(f" - Total Test Cases:   {total_test_frames}")
    print(f" - Correct Detections: {passed_frames}")
    print(f" - Final Accuracy:     {success_rate * 100:.2f}%")
    print("-" * 57)

    if success_rate >= success_threshold:
        print("✅ All tests passed successfully!")
        print("   Turret Visual Servoing implementation is correct.")
    else:
        print("❌ Implementation failed. Accuracy is below the required 90%.")
        if fail_details:
            print("\n   First few errors found:")
            for err in fail_details:
                print(f"   -> {err}")
    print("="*57 + "\n")

if __name__ == "__main__":
    run_autograder()