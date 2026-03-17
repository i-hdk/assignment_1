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
    
    # Updated test cases: only using the 6 new images
    test_cases = [
        ("objects_detected_1_1.png", 1, "sports ball", [(0.752857506275177, (801.5690307617188, 325.81195068359375))]),
        ("objects_detected_1_2.png", 1, "banana", [(0.6973349452018738, (676.2200317382812, 512.9927368164062))]),
        ("objects_detected_2_1.png", 2, "sports ball", [
            (0.8348134756088257, (781.9148559570312, 312.924560546875)),
            (0.6091908812522888, (492.20159912109375, 256.128173828125)),
            (0.2640124261379242, (880.47265625, 472.1309509277344))
        ]),
        ("objects_detected_2_2.png", 2, "banana", []), # there are 3 sports balls here, but no bananas.
        ("objects_detected_3_1.png", 3, "frisbee", [
            (0.6725870370864868, (587.9255981445312, 499.11370849609375))
        ]),
        ("objects_detected_3_2.png", 3, "sports ball", [
            (0.7930213809013367, (579.3873901367188, 356.0917663574219)),
            (0.2538105845451355, (154.5794677734375, 472.85614013671875))
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
                    print(f"❌ FAIL: Found confidence {results[i][0]} object, but expected {expected_results[i][0]}.")

                if abs(results[i][1][0] - expected_results[i][1][0]) < 0.001:
                    print(f"✅ PASS: Found result with expected x={expected_results[i][0]}")
                else:
                    print(f"❌ FAIL: Found x={results[i][0]}, but expected {expected_results[i][0]}.")

                if abs(results[i][1][1] - expected_results[i][1][1]) < 0.001:
                    print(f"✅ PASS: Found result with expected y={expected_results[i][1]}")
                else:
                    print(f"❌ FAIL: Found y={results[i][1]}, but expected {expected_results[i][1]}.")

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