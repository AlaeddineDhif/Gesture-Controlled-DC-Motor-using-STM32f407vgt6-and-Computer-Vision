import cv2
import mediapipe as mp
import serial
import time

# MediaPipe Configuration
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,  # Only one hand detected
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

# Serial Configuration
ser = serial.Serial('COM6', 115200, timeout=1)  # Adjust the COM port
time.sleep(2)  # Initialization

# Camera Configuration
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


def count_fingers(landmarks, hand_type):
    """
    Counts the number of raised fingers with left/right hand handling.
    :param landmarks: The detected hand landmarks.
    :param hand_type: "Left" or "Right" to adapt the thumb logic.
    :return: Number of raised fingers (0 to 5).
    """
    tip_ids = [4, 8, 12, 16, 20]  # IDs of the finger tips
    mcp_ids = [2, 5, 9, 13, 17]  # IDs of the MCP joints
    fingers = 0

    # Thumb (adaptation for left/right hand)
    if hand_type == "Right":
        # Right hand: the thumb is to the left of the MCP when raised
        if landmarks.landmark[tip_ids[0]].x < landmarks.landmark[mcp_ids[0]].x:
            fingers += 1
    else:
        # Left hand: the thumb is to the right of the MCP when raised
        if landmarks.landmark[tip_ids[0]].x > landmarks.landmark[mcp_ids[0]].x:
            fingers += 1

    # Other fingers (Y comparison, valid for both hands)
    for i in range(1, 5):
        if landmarks.landmark[tip_ids[i]].y < landmarks.landmark[mcp_ids[i]].y:
            fingers += 1

    return fingers


def get_motor_state(num_fingers):
    """
    Returns the motor state based on the number of fingers.
    :param num_fingers: Number of raised fingers (0 to 5).
    :return: Motor state (str).
    """
    if num_fingers == 0:
        return "Stop"
    elif num_fingers == 1:
        return "Speed 1"
    elif num_fingers == 2:
        return "Speed 2"
    elif num_fingers == 3:
        return "Speed 3"
    elif num_fingers == 4:
        return "Speed 4"
    elif num_fingers == 5:
        return "Speed 5"
    else:
        return "Unknown"


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Image processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Default variables
    num_fingers = 0
    hand_type = "None"
    motor_state = "Stop"
    motor_direction = "None"

    if results.multi_hand_landmarks:
        try:
            # Take the first detected hand
            hand = results.multi_hand_landmarks[0]
            handedness = results.multi_handedness[0].classification[0].label  # "Left" or "Right"
            num_fingers = count_fingers(hand, handedness)  # Count fingers
            hand_type = handedness

            # Determine motor state
            motor_state = get_motor_state(num_fingers)

            # Determine rotation direction
            if hand_type == "Right":
                motor_direction = "Clockwise"
            elif hand_type == "Left":
                motor_direction = "Counterclockwise"

            # Draw landmarks with bright colors
            mp_drawing.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),  # Red points
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)   # Green connections
            )

            # Draw a green frame around the hand
            x_coords = [int(landmark.x * frame.shape[1]) for landmark in hand.landmark]
            y_coords = [int(landmark.y * frame.shape[0]) for landmark in hand.landmark]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            cv2.rectangle(frame, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20),
                          (0, 255, 0), 2)  # Green frame around the hand

        except IndexError:
            pass

    # Semi-transparent background for text (increased width)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (350, 90), (0, 0, 0), -1)  # Black rectangle (increased width)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Display information (adjusted text)
    cv2.putText(frame, f"State: {motor_state}", (20, 30),  # Adjusted position
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)  # Reduced font size
    cv2.putText(frame, f"Direction: {motor_direction}", (20, 55),  # Adjusted position
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)  # Reduced font size
    cv2.putText(frame, f"Hand: {hand_type}", (20, 80),  # Adjusted position
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)  # Reduced font size

    # Serial send (number of fingers and hand type)
    command = f"{num_fingers},{hand_type}\n"
    ser.write(command.encode('ascii'))

    # Display window
    cv2.imshow('Hand Control - Motor Speed', frame)

    # Quit with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
ser.close()