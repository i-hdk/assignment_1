# 16-180 - Concepts of Robotics - Assignment 1

Due Date: Feb. 2, 2026, 11:59 PM

## Setup

In VS Code, open your `16-180_Concepts_of_Robotics` folder if it is not already open. Open a terminal and run the following:
```
git clone https://github.com/cmu-16-180/assignment_1.git
```
You should see an `assignment_1` folder where you do your work.

 Note: if you are familiary with `cd`, do not use it. All assignment code is meant to be run from the `16-180_Concepts_of_Robotics` folder.

**Important!** When working on this assignment, make sure that the virtual environment is active. You should see `(venv)` at the beginning of each prompt line on your terminal. See the setup guide if it is not active.

## Problem 1 (25 Points): CMU Robotics Self-Guided Tour 

There are some places on comapus where you can learn about the history of robotics at CMU. The [Robotics Project](https://library.cmu.edu/university-archives/initiatives/robotics-project) has a website archive and a physical exhibit in the Hunt Library. You can also wander the hallways of Newell Simon Hall, where you will find many photos of CMU robots, and even some old robots and components on display.  

As you explore these places, take **selfies with at least three different robots** that have distinct applications. For each robot, describe its design, application, and how sensing contributes to its operation. Identify the sensors in each robot and explain their role in enabling the robot's functionality.

After writing about all the robots you encountered, conclude your essay by choosing your favorite one and explaining why it stands out to you, and provide thoughtful insights based on your observations and research.

Your essay for all parts above should be (300 - 500 words).

## Problem 2 (25 Points): Dead Reckoning

In this problem, you will implement dead reckoning to estimate the position of a differential drive robot given measurements from the wheel encoders.  The robot consists of two drive wheels on a single axis and a passive caster wheel for balance. The key parameters of the robot are as follows:

- **Wheel radius:** 0.1 m
- **Distance between the centers of the two drive wheels:** 0.2 m
- **Encoder resolution:** 4096 ticks per revolution

Do you work in `assignment_1/dead_reckoning_submission.py`.

To get started, run:
* Windows: `python assignment_1\dead_reckoning_main.py`
* MacOS: `mjpython assignment_1/dead_reckoning_main.py`

You should see two differential drive cars. One will move, and the other will stay still. The one that moves is the "real" robot's motion. The one that does not move is your computed **dead reckoning** estimate of where the robot actually is--your job is to compute this based on the wheel encoder readings.

### Part 1:

Each wheel has an encoder. The driver provides you with an array of the accumulated number of ticks at each timestep for each wheel. Your task is to convert these accumulated tick values to a forward velocity (m/s) at each timestep and an angular velocity (rad/s) at each timestep. Do this in the `ticks2vel` method.

### Part 2:

Once your `ticks2vel` method is implemented, your next task is to convert these values to an (x,y,angle) position at each time step. For this, you will implement the `dead_reckoning_from_encoders` method.

### Autograder

After you're happy with your implementation, run the autograder to check that you haven't missed anything critical. This is the same autograder tests that we will run for grading.
* Windows `python assignment_1\dead_reckoning_autograder.py`
* MacOS `python assignment_1/dead_reckoning_autograder.py` (Note: you do **not** need mjpython for the autograder since it does not use Mujoco)

## Problem 3 (25 Points): RGB Color Segmentation

In this problem, you will build the "eyes" for a robotic search-and-rescue turret. The robot is mechanically capable of moving (pan/tilt), but it is currently blind. Your job is to write a computer vision algorithm that locates a specific "Beacon" in a cluttered environment so the robot can lock onto it.

### Concepts Covered:
* Computer Vision (RGB vs. HSV color spaces).
* Thresholding and Binary Masks.
* Visual Servoing (controlling a robot based on camera feedback).

### The Scenario
You are provided with a simulation of a Pan-Tilt Turret equipped with a camera.

* **The Target:** A Glowing Red Sphere (The Beacon).
* **The Distractors:** The environment contains "false positives" to trick your algorithm:
    * A Dark Red Cube (Correct Hue, Wrong Value/Shape).
    * An Orange Sphere (Wrong Hue, Right Shape).
* **The Goal:** Return the (x, y) pixel coordinates of the Beacon's center. If your code works, the instructor's control loop will automatically rotate the turret to keep the target centered.


### Run the Simulation
Open VS Code, activate your virtual environment, and run:
* Windows/Linux: `python assignment_1\turret_main.py`
* MacOS: `mjpython assignment_1/turret_main.py`

What to Expect
* **Two Windows:** You should see two windows appear:
    1. **MuJoCo Viewer:** The main 3D simulation showing the robot and the room.
    2. **Robot Camera Feed (Debug Process):** A smaller window showing exactly what the robot sees.
* **Windows Users:** The first time you run this, you may get a **Windows Firewall** popup asking if Python can access the network.
    * **Action:** Click **"Allow Access".** (This is required because the two windows talk to each other using internal network sockets).
* **Initial Behavior:** The robot will likely stare at the center of the room or drift aimlessly. This is normal! You haven't written the vision code yet.

### Your Task
Open `assignment_1/turret_submission.py` in VS Code, and edit the function `find_target(image)`.
```
def find_target(image):
    # image is a (480, 640, 3) numpy array of RGB pixels
    ...
    return (cx, cy)
```
#### Implementation Strategy:
1. Color Space Conversion: The raw image is in RGB. It is often hard to detect "Red" in RGB because lighting changes the values. Converting to HSV (Hue, Saturation, Value) makes it much easier to isolate colors.
    * *Hint:* Use `cv2.cvtColor`.
2. Thresholding: Create a "mask" (a black and white image) where white pixels represent the target color and black pixels are everything else.
    * *Hint:* You might need two masks because "Red" in HSV wraps around the hue circle (both 0-10 and 170-180 are red).
3. Filtering: Eliminate the distractors.
    * The Dark Red Cube has the right color but is very dark (Low Value).
    * The Orange Sphere is bright but has the wrong Hue.
4. Centroid Calculation: Once you have a clean mask, calculate the center of the white blob.
    * *Hint:* Use `cv2.moments`.
    
### Success Criteria

You know you are done when:

1. **Green Crosshair:** In the "Robot Camera Feed" window, a green circle stays locked onto the Red Sphere.
2. **Tracking:** The physical robot in the main window smoothly rotates to follow the sphere as it circles the room.
3. **Rejection:** The robot ignores the Red Cube and Orange Sphere, even when they cross the camera's view.

### Interaction & Shutdown
* **To Quit:** Press **ESC** in the Camera Window or close the MuJoCo window. Both will shut down the program cleanly.
* **Troubleshooting:**
    * *The robot shakes violently:* Your vision code might be jumping between the target and a distractor. Check your thresholds.
    * *The Camera Window is black/frozen:* Check the terminal for errors. Ensure you aren't stuck in an infinite loop inside find_target.

### Autograder

After you're happy with your implementation, run the autograder to check that you haven't missed anything critical. This is the same autograder tests that we will run for grading.
* Windows `python assignment_1\turret_autograder.py`
* MacOS `python assignment_1/turret_autograder.py`

## Problem 4 (25 Points): YOLO searching

YOLO (You Only Look Once) is a very popular neural-network based object detection system. Simply put: you provide it an image, and it gives you an estimate (or "prediction") of where objects are. There is a big caveat though: it only knows how to find objects that were in its training set. 

What you're going to do is use YOLO to find objects of interest (oddly, "sports ball" are of main interest, but not the only thing). You will download a trained YOLO model, and use it on an image, then process the results.

Run this:
* Windows: `python assignment_1\yolo_main.py`
* MacOS: `mjpython assignment_1/yolo_main.py`

What you should see is a bunch of random object fall in front of the camera.

### Step 1: run the YOLO model on the image

Update `yolo_submission.py` to run YOLO on the model, an then ask it to annotate the image for you. Once you return the annotated image, you'll see detection boxes with class names and confidence levels on them.

### Step 2: find the target object

Update your solution to search through the YOLO results to find the object of interest (see `yolo_submission.py` comments for details). While testing your implementation, we recommend printing the results you find. Your final task is to return the results so that the autograder passes.

After you're happy with your implementation, run the autograder to check that you haven't missed anything critical. This is the same autograder tests that we will run for grading.
* Windows: `python assignment_1\yolo_autograder.py`
* MacOS: `mjpython assignment_1/yolo_autograder.py`