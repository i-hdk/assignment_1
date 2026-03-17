import mujoco
import mujoco.viewer
import numpy as np
import time
import cv2 
import multiprocessing
import queue 
import turret_submission as student

# -----------------------------------------------------------------------------
# SEPARATE PROCESS: GUI WORKER
# -----------------------------------------------------------------------------
def run_debug_window(image_queue, result_queue, stop_event):
    """
    Runs in a separate process. Loops until 'stop_event' is set.
    """
    window_name = "Robot Camera Feed"

    student.enable_visualization()
    
    try:
        # Create window explicitly to allow property checking
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        while not stop_event.is_set():
            try:
                # Wait for a new frame with a VERY short timeout
                # This ensures we check for ESC/Close events rapidly
                frame = image_queue.get(timeout=0.01)

                cx, cy = student.find_target(frame)
                
                # Convert RGB (MuJoCo) -> BGR (OpenCV)
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                if cx is not None and cy is not None:
                    result_queue.put((cx, cy))
                    cv2.circle(bgr_frame, (int(cx), int(cy)), 10, (0, 255, 0), 2)

                # Show the window
                cv2.imshow(window_name, bgr_frame)
                
                # CHECK 1: Process Window Events
                # Pressing ESC (key 27) will exit
                key = cv2.waitKey(1) & 0xFF
                if key == 27: 
                    print("User pressed ESC in Debug Window. Stopping...")
                    stop_event.set() # Signal main process to stop
                    break
                
                # CHECK 2: Detect if user clicked 'X' on the window
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("User closed Debug Window. Stopping...")
                    stop_event.set() # Signal main process to stop
                    break
                    
            except queue.Empty:
                # No frame received recently.
                # Crucial: We must still pump the cv2 event loop or the window freezes
                cv2.waitKey(1) 
                pass
            except KeyboardInterrupt:
                stop_event.set()
                break
                
    except Exception as e:
        print(f"Debug Window Error: {e}")
    finally:
        try:
            cv2.destroyAllWindows()
        except:
            pass

# -----------------------------------------------------------------------------
# INSTRUCTOR SECTION: SIMULATION LOOP
# -----------------------------------------------------------------------------
def main():
    # --- CONFIGURATION ---
    ENABLE_VISION_DEBUG = True 

    # --- MULTIPROCESSING SETUP ---
    debug_process = None
    image_queue = None
    result_queue = None
    stop_event = None
    
    if ENABLE_VISION_DEBUG:
        image_queue = multiprocessing.Queue(maxsize=2)
        result_queue = multiprocessing.Queue(maxsize=2)
        stop_event = multiprocessing.Event()
        
        # Pass the stop_event to the child process
        debug_process = multiprocessing.Process(
            target=run_debug_window, 
            args=(image_queue, result_queue, stop_event)
        )
        debug_process.daemon = True 
        debug_process.start()
        print("Debug Window: LAUNCHED (Separate Process)")

    xml_path = "assignment_1/assets/turret.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    renderer = mujoco.Renderer(model, height=480, width=640)

    print("Running Assignment 1 (Position Control Mode)...")
    
    current_pan = 0.0
    current_tilt = 0.0
    frame_count = 0
    
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            start_time = time.time()
            
            # Loop runs as long as:
            # 1. MuJoCo window is open
            # 2. Stop event hasn't been triggered by the Debug Window (ESC)
            while viewer.is_running() and (stop_event is None or not stop_event.is_set()):
                step_start = time.time()
                frame_count += 1

                # 1. SIMULATE WORLD
                t = time.time() - start_time
                data.qpos[2] = 2.0 + 0.5 * np.cos(0.5 * t) 
                data.qpos[3] = 1.0 * np.sin(0.5 * t)       
                
                mujoco.mj_step(model, data)

                # 2. CAPTURE IMAGE
                renderer.update_scene(data, camera="eye")
                rgb_image = renderer.render()

                # 3. RUN STUDENT CODE
                if debug_process.is_alive():
                    if not image_queue.full():
                        debug_image = rgb_image.copy()
                        image_queue.put(debug_image)
                else:
                    stop_event.set()

                # 4. VISUAL SERVOING
                try:
                    cx, cy = result_queue.get_nowait()

                    screen_center_x = 640 / 2
                    screen_center_y = 480 / 2
                    
                    error_x = (screen_center_x - cx)
                    error_y = (screen_center_y - cy)

                    current_pan += 0.0005 * error_x
                    current_tilt += 0.0005 * error_y

                    current_pan = np.clip(current_pan, -1.5, 1.5)
                    current_tilt = np.clip(current_tilt, -0.5, 0.5)
                except queue.Empty:
                    pass

                data.ctrl[0] = current_pan
                data.ctrl[1] = current_tilt

                # # 5. SEND TO DEBUG WINDOW
                # if ENABLE_VISION_DEBUG and (frame_count % 2 == 0):
                #     if debug_process.is_alive():
                #         if not image_queue.full():
                #             debug_image = rgb_image.copy()
                #             if cx is not None and cy is not None:
                #                 cv2.circle(debug_image, (int(cx), int(cy)), 10, (0, 255, 0), 2)
                            
                #             image_queue.put(debug_image)
                #     else:
                #         # Process died unexpectedly
                #         stop_event.set()

                # 6. RENDER SIMULATION
                viewer.sync()
                
                # Timing
                time_until_next = 0.016 - (time.time() - step_start)
                if time_until_next > 0:
                    time.sleep(time_until_next)
                    
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        # CLEANUP
        if stop_event:
            stop_event.set()
        
        # PREVENT HANGS:
        # If the queue has data, the process might hang trying to flush it.
        # cancel_join_thread tells the process "don't wait for the queue"
        if image_queue:
            image_queue.cancel_join_thread()
            image_queue.close()

        if debug_process:
            debug_process.join(timeout=1.0)
            if debug_process.is_alive():
                print("Force terminating debug process...")
                debug_process.terminate()
            
        print("Cleanup complete.")

if __name__ == "__main__":
    main()