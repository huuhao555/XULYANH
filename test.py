import cv2

cap = cv2.VideoCapture(0)

text1 = "XU LY ANH VA THI GIAC MAY TINH - 010100086901"
text2 = "NHOM 7"

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.flip(image, 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    text_color = (255, 255, 255)
    thickness = 2
    bg_color = (0 ,178, 238)

    height, width, _ = image.shape

    (text1_width, text1_height), _ = cv2.getTextSize(text1, font, font_scale, thickness)
    (text2_width, text2_height), _ = cv2.getTextSize(text2, font, font_scale, thickness)

    text1_x = (width - text1_width) // 2
    text1_y = 50

    text2_x = (width - text2_width) // 2
    text2_y = text1_y + text1_height + 20

    padding = 10
    cv2.rectangle(image, 
                  (text1_x - padding, text1_y - text1_height - padding), 
                  (text1_x + text1_width + padding, text1_y + padding), 
                  bg_color, -1)

    cv2.rectangle(image, 
                  (text2_x - padding, text2_y - text2_height - padding), 
                  (text2_x + text2_width + padding, text2_y + padding), 
                  bg_color, -1)
    
    cv2.putText(image, text1, (text1_x, text1_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
    cv2.putText(image, text2, (text2_x, text2_y), font, font_scale, text_color, thickness, cv2.LINE_AA)

    cv2.imshow('Camera NHOM 7', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
