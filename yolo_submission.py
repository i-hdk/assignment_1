import cv2
import os

from ultralytics import YOLO

yolo_model = None

def load_yolo_model():
    global yolo_model
    if yolo_model is None:
        print("Loading pre-trained YOLOv8 model...")
        print("(Downloading model if not already present...)")
        try:
            model_path = os.path.join(os.path.dirname(__file__), 'assets', 'yolov8n.pt')
            yolo_model = YOLO(model_path)  # Automatically downloads if not present
            print("✓ Model loaded successfully!")
            # help(yolo_model.__call__)
            # print(yolo_model)
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            print("Make sure you have an internet connection to download the model.")
            raise
    return yolo_model

def detect_objects(img, target_name="sports ball"):
    """
    Render the scene, detect objects using YOLO, and display the results.
    
    Args:
        renderer: Mujoco renderer
        data: Mujoco data
    """
    # Render from camera for YOLO detection
    
    # load the yolo model    
    yolo_model = load_yolo_model()
    
    # TODO: Use the yolo_model to run YOLO detection
    
    yolo_results = yolo_model(img) # TODO: YOLO Model results on img

    # TODO: once you have this running, make sure you return it, you'll
    # start seeing the results in the window.
    annotated_img = yolo_results[0].plot() # TODO: results rendered on img

    # TODO: process results, look for a detections with the class name 
    # target_name (default "sports ball")
    # Fill in search_results with the confidence, and box center
    # E.g., if there is a "sports ball" with confidence 0.23 centered
    # at pixel (50, 60), the search_results should contain an entry
    #    (0.23, (50, 60))
    # Thus if you had multiple search results, you might end up with
    # a search_results that looks like this:
    #    search_results = [
    #        (0.23, (50, 60)),
    #        (0.75, (23, 44)),
    #         ...
    #    ]
    search_results = []

    for r in yolo_results:
        print(r.names)
        print(r.boxes.xyxy)
        print(r.boxes.cls)
        i = 0
        for idx in r.boxes.cls :
            print("idx is ")
            print(idx)
            print(r.names.get (int(idx)) )
            if r.names.get (int(idx))  == target_name :
                (x1,y1,x2,y2) = r.boxes.xyxy[i]
                search_results.append((r.boxes.conf, ((x1+y1)/2.0, (x2+y2)/2.0)))
            i+=1
    
    #    # see https://docs.ultralytics.com/modes/predict/#boxes for details
    #    # Note: r.names is a list that is indexed by result class
    #    r.boxes.cls is a list of all class indexes (they're floats, so you'll need to cast to int to index into r.names)
    #    r.boxes.conf is a list of all confidences levels
    #    r.boxes.xyxy is a list of all boxes in xyxy format
    #    r.boxes.xywh is a list of all boxes in xywh format
    #    Hint: use search_results.append((conf, (x, y))) once you have it

    
    # return annotated_img, results
    return annotated_img, search_results # TODO: replace with your return value

