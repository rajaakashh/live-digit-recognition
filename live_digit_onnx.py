import cv2
import numpy as np
import onnxruntime as ort

# -------------------------------
# 1. Load ONNX model
# -------------------------------
session = ort.InferenceSession("mnist.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# -------------------------------
# 2. Start webcam
# -------------------------------
cap = cv2.VideoCapture(0)
print("Press SPACE to predict | Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # ROI box
    x1, y1, x2, y2 = 200, 100, 450, 350
    roi = frame[y1:y2, x1:x2]
    roi = cv2.flip(roi, 1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(
        frame,
        "Write digit inside box, press SPACE",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.imshow("Live Digit Recognition", frame)

    key = cv2.waitKey(1) & 0xFF

    # -------------------------------
    # 3. Capture & predict
    # -------------------------------
    if key == ord(' '):
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Threshold
        _, thresh = cv2.threshold(
            blur, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            print("No digit detected")
            continue

        # Largest contour = digit
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        digit = thresh[y:y+h, x:x+w]

        # Resize while keeping aspect ratio (MNIST style)
        h, w = digit.shape
        if h > w:
            new_h = 20
            new_w = int(w * (20 / h))
        else:
            new_w = 20
            new_h = int(h * (20 / w))

        digit = cv2.resize(digit, (new_w, new_h))

        # Pad to 28x28
        canvas = np.zeros((28, 28), dtype=np.uint8)
        x_offset = (28 - new_w) // 2
        y_offset = (28 - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit

        # Normalize
        img = canvas.astype(np.float32) / 255.0
        img = img.reshape(1, 1, 28, 28)

        # Inference
        output = session.run(None, {input_name: img})
        prediction = int(np.argmax(output[0]))

        print("Predicted Digit:", prediction)

        # Show processed image
        cv2.imshow("Processed 28x28", canvas)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
