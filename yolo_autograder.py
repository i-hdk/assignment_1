import cv2
import os
import numpy as np
import yolo_submission

class MockRenderer:
    """Fakes the Mujoco Renderer to return static test images."""
    def __init__(self, image_path):
        self.image_path = image_path

    def update_scene(self, data, camera):
        # Do nothing, as there is no real Mujoco scene
        pass

    def render(self):
        # Load the specific test image requested for this test case
        img = cv2.imread(self.image_path)
        if img is None:
            raise FileNotFoundError(f"Could not find test image at {self.image_path}")
        return img

def run_tests():
    test_folder = "assignment_1/assets/test_yolo"
    
    test_cases = [
        ("objects_detected_1_1.png", 1, "sports ball", [(0.7528566122055054, (782.3972778320312, 307.33746337890625))]),
        ("objects_detected_1_2.png", 1, "banana", [(0.6973350048065186, (615.2167358398438, 469.90460205078125))]),
        ("objects_detected_2_1.png", 2, "sports ball", [
            (0.8348131179809570, (737.8557739257812, 269.03033447265625)),
            (0.6091901063919067, (459.68328857421875, 242.8646240234375)),
            (0.2640120983123779, (825.5781860351562, 421.7991943359375))
        ]),
        ("objects_detected_2_2.png", 2, "banana", []),  # there are 3 sports balls here, but no bananas.
        ("objects_detected_3_1.png", 3, "frisbee", [
            (0.6725848317146301, (526.8341064453125, 456.82562255859375))
        ]),
        ("objects_detected_3_2.png", 3, "sports ball", [
            (0.7930208444595337, (554.5283813476562, 337.42437744140625)),
            (0.2538095712661743, (117.69468688964844, 434.85260009765625))
        ])
    ]

    total_passed = 0
    total_tests = len(test_cases)

    print("--- Starting Gradescope-style YOLO Autograder ---")
    
    # Pre-load the model once
    yolo_submission.load_yolo_model()

    for filename, min_expected, search_term, expected_results in test_cases:
        image_path = os.path.join(test_folder, filename)
        
        if not os.path.exists(image_path):
            print(f"⚠️ SKIPPING: {filename} not found in {test_folder}")
            continue

        print(f"\nTesting {filename} (Expecting at {len(expected_results)} detections)...")
        
        try:
            # Create the mock renderer with the specific image
            mock_renderer = MockRenderer(image_path)
            
            # Call the user's function
            _, results = yolo_submission.detect_objects(mock_renderer.render(), search_term)
            
            # Get detection count from YOLO results
            detections = len(results)
            
            if detections == len(expected_results):
                print(f"✅ PASS: Found {detections} objects.")
                total_passed += 1
            else:
                print(f"❌ FAIL: Found only {detections} objects, but expected {len(expected_results)}.")

            results.sort()
            expected_results.sort()
            for i in range(detections):
                if abs(results[i][0] - expected_results[i][0]) < 0.001:
                    print(f"✅ PASS: Found result with expected confidence {expected_results[i][0]}")
                else:
                    print(f"❌ FAIL: Found confidence {results[i][0]}, but expected {expected_results[i][0]}.")

                if abs(results[i][1][0] - expected_results[i][1][0]) < 0.001:
                    print(f"✅ PASS: Found result with expected x={expected_results[i][1][0]}")
                else:
                    print(f"❌ FAIL: Found x={results[i][1][0]}, but expected {expected_results[i][1][0]}.")

                if abs(results[i][1][1] - expected_results[i][1][1]) < 0.001:
                    print(f"✅ PASS: Found result with expected y={expected_results[i][1][1]}")
                else:
                    print(f"❌ FAIL: Found y={results[i][1][1]}, but expected {expected_results[i][1][1]}.")

        except Exception as e:
            print(f"⚠️ ERROR during test for {filename}: {e}")

    # Final result reporting
    print("\n---------------------------------------------------------")
    if total_passed == total_tests and total_tests > 0:
        print("All test passed successfully and YOLO detection implementation is correct!")
    elif total_tests == 0:
        print("Error: No test cases found.")
    else:
        print(f"Test complete. {total_passed}/{total_tests} cases passed.")
    print("---------------------------------------------------------")
    
    return total_passed == total_tests

if __name__ == "__main__":
    run_tests()