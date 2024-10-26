import cv2
import mediapipe as mp
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


ret, frame = cap.read()
if not ret:
    print("Error: Could not read frame from webcam.")
    cap.release()
    exit()

frame_height, frame_width = frame.shape[:2]


drawing_image = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)


prev_x, prev_y = None, None


current_color = (0, 0, 255)


color_dict = {
    'Red': (0, 0, 255),
    'Green': (0, 255, 0),
    'Blue': (255, 0, 0),
    'Yellow': (0, 255, 255)
}

def check_if_hand_open(hand_landmarks, image_shape):
    """Check if all five fingers are open."""
    image_height, image_width = image_shape[:2]
    fingers_tips_ids = [
        mp_hands.HandLandmark.THUMB_TIP,
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]

    fingers_tips = [hand_landmarks.landmark[tip].y * image_height for tip in fingers_tips_ids]
    fingers_mcp = [
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * image_height,
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height,
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height,
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height,
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height
    ]


    return all(tip < mcp for tip, mcp in zip(fingers_tips, fingers_mcp))


with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

    
        drawing_image = cv2.resize(drawing_image, (image.shape[1], image.shape[0]))

    
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    
        results = hands.process(image_rgb)

    
        image.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=4, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=4, circle_radius=2)
                )

            
                if check_if_hand_open(hand_landmarks, image.shape):
                    drawing_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
                
                    prev_x, prev_y = None, None
                    continue

            
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                x1 = int(index_finger_tip.x * image.shape[1])
                y1 = int(index_finger_tip.y * image.shape[0])
                x2 = int(thumb_tip.x * image.shape[1])
                y2 = int(thumb_tip.y * image.shape[0])

            
                if abs(x1 - x2) > 50 and abs(y1 - y2) > 50 and y2 > y1 and x2 > x1:
                
                    if prev_x is not None and prev_y is not None:
                        cv2.line(drawing_image, (prev_x, prev_y), (x1, y1), current_color, 5)

                
                    prev_x, prev_y = x1, y1
                else:
                
                    prev_x, prev_y = None, None
        else:
        
            prev_x, prev_y = None, None

    
        key = cv2.waitKey(1)
        if key != 255:
            if key == ord('r') or key == ord('R'):
                current_color = (0, 0, 255)
                print("Color changed to Red")
            elif key == ord('g') or key == ord('G'):
                current_color = (0, 255, 0)
                print("Color changed to Green")
            elif key == ord('b') or key == ord('B'):
                current_color = (255, 0, 0)
                print("Color changed to Blue")
            elif key == ord('y') or key == ord('Y'):
                current_color = (0, 255, 255)
                print("Color changed to Yellow")
            elif key == 27:
                break

    
        combined_image = cv2.addWeighted(image, 0.7, drawing_image, 0.3, 0)

    
        combined_image = cv2.flip(combined_image, 1)

    
        text1 = "Xu ly anh va thi giac may tinh - 010100086901"
        text2 = "Nhom 7sss"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2

    
        text1_size = cv2.getTextSize(text1, font, font_scale, font_thickness,)[0]
        text2_size = cv2.getTextSize(text2, font, font_scale, font_thickness)[0]

        text1_x = (combined_image.shape[1] - text1_size[0]) // 2
        text1_y = 50

        text2_x = (combined_image.shape[1] - text2_size[0]) // 2
        text2_y = text1_y + 40

    
        cv2.putText(combined_image, text1, (text1_x, text1_y), font, font_scale, (255, 255, 255), font_thickness)
        cv2.putText(combined_image, text2, (text2_x, text2_y), font, font_scale, (255, 255, 255), font_thickness)

    
        cv2.imshow('BAN TAY TA LAM NEN TAT CA', combined_image)


cap.release()
cv2.destroyAllWindows()
