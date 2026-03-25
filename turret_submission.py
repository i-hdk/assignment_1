import cv2 
import numpy as np
import matplotlib.pyplot as plt

hsv_window_name = "HSV Visualizer"
visual_on = False

def enable_visualization():
    """This method is called by main_turret.py, but not autograder_turret.py."""
    global visual_on
    if not visual_on:
        visual_on = True
        cv2.namedWindow(hsv_window_name, cv2.WINDOW_AUTOSIZE)
        print("Visualizations enabled")

def visualize_hsv(hsv):
    if visual_on:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        color = (255,255,255)
        thick = 1
        h_channel, s_channel, v_channel = cv2.split(hsv)
        h_channel = cv2.cvtColor(h_channel, cv2.COLOR_GRAY2BGR)
        s_channel = cv2.cvtColor(s_channel, cv2.COLOR_GRAY2BGR)
        v_channel = cv2.cvtColor(v_channel, cv2.COLOR_GRAY2BGR)
        cv2.putText(h_channel, "Hue", (10,25), font, scale, color, thick)
        cv2.putText(s_channel, "Saturation", (10,25), font, scale, color, thick)
        cv2.putText(v_channel, "Value", (10,25), font, scale, color, thick)
        stacked_hsv = np.hstack((h_channel, s_channel, v_channel))
        cv2.imshow(hsv_window_name, stacked_hsv)

def visualize_mask(mask_name, mask):
    if visual_on:
        cv2.namedWindow(mask_name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(mask_name, mask)

# -----------------------------------------------------------------------------
# STUDENT SECTION: EDIT THIS FUNCTION
# -----------------------------------------------------------------------------
def find_target(imageRGB):
    """
    Process the image to find the target (Glowing Red Sphere).
    
    Args:
        image: A numpy array of shape (Height, Width, 3) representing RGB pixels.
    
    Returns:
        (cx, cy): The (x, y) pixel coordinates of the center of the target.
                  Return (None, None) if no target is found.
    """
    # ------------------ YOUR CODE HERE ------------------
    # 1. Define color range for the "Glowing Red" target
    # 2. Threshold the image
    # 3. Find contours or centroids (moments)
    # 4. Return the center
    
    # Convert from RGB to HSV for better color segmentation
    hsv = cv2.cvtColor(imageRGB, cv2.COLOR_RGB2HSV)

    # Optional, you can visualize the HSV images with this method:
    # visualize_hsv(hsv)

    # 1. Define color range for the glowing ball target
    # TODO: set the lower and upper ranges to select target pixels
    lower_red = np.array([170, 250, 245])  # Lower bound of red in HSV
    upper_red = np.array([179, 255, 255])  # Upper bound of red in HSV

    lower_red2 = np.array([0, 240, 240])  # Lower bound of red in HSV
    upper_red2 = np.array([10, 255, 255])  # Upper bound of red in HSV

    # 3. Threshold the image using OpenCV inRange method
    mask1 = cv2.inRange(hsv, lower_red, upper_red) # TODO: use the OpenCV inRange method to create a binary image based on thresholds lower_red and upper_red
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2) # TODO: use the OpenCV inRange method to create another binary from the second set of thresholds

    # Combine masks (note: you can create more masks and combine them with the | operator here)
    mask = mask1 | mask2

    # Optional, you can visualize your masks with this method:
    # visualize_mask("First Mask", mask1)
    # visualize_mask("Combined Mask", mask)

    contours = None
    # 3. Find contours in the mask
    # Once your masks are defined, uncomment the following:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            # 4. return the center
            return (cx, cy)
        
    return (None, None)