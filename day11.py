import cv2
import matplotlib.pyplot as plt

# Load image
img_path = r"images\sample.jpg"  # Make sure this file exists
img = cv2.imread(img_path)

# Check if image loaded
if img is None:
    print("Error: Image not found!")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 1️⃣ Apply Gaussian Blur
blur_gaussian = cv2.GaussianBlur(gray, (5,5), 0)

# 2️⃣ Apply Median Blur
blur_median = cv2.medianBlur(gray, 5)

# 3️⃣ Canny Edge Detection
edges = cv2.Canny(gray, 100, 200)

# Display results
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(blur_gaussian, cmap='gray')
plt.title("Gaussian Blur")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(edges, cmap='gray')
plt.title("Canny Edges")
plt.axis('off')

plt.tight_layout()
plt.show()
