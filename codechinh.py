import cv2
import numpy as np

# Load the original image
image_path = r'D:\HK5\CKXLAS\Anhtest1\Screenshot 2024-01-01 201729.png'
image = cv2.imread(image_path)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

# Define range of RED color in HSV
lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])
lower_red2 = np.array([170, 50, 50])
upper_red2 = np.array([180, 255, 255])

# Threshold the HSV image to get only red colors
mask1 = cv2.inRange(hsv, lower_red, upper_red)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = mask1 + mask2

# Bitwise-AND mask and original image
red_hue_image = cv2.bitwise_and(image, image, mask=mask)

# Convert to grayscale
gray_red_hue = cv2.cvtColor(red_hue_image, cv2.COLOR_BGR2GRAY)

# Function to calculate LBP value for a pixel
def calculate_lbp_pixel(img, x, y):
    center = img[x, y]
    values = []

    # Define the neighborhood points
    points = [(x-1, y-1), (x-1, y), (x-1, y+1),
              (x, y+1), (x+1, y+1), (x+1, y),
              (x+1, y-1), (x, y-1)]

    for point_x, point_y in points:
        values.append(1 if img[point_x, point_y] >= center else 0)

    # Convert binary values to decimal
    lbp_value = sum([val * (2 ** idx) for idx, val in enumerate(values)])

    return lbp_value

# Apply LBP to the entire image
rows, cols = gray_red_hue.shape
lbp_result = np.zeros_like(gray_red_hue, dtype=np.uint8)

for i in range(1, rows - 1):
    for j in range(1, cols - 1):
        lbp_result[i, j] = calculate_lbp_pixel(gray_red_hue, i, j)

# Use Hough Circle Detection
circles = cv2.HoughCircles(
    gray_red_hue,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=50,
    param1=50,
    param2=30,
    minRadius=30,
    maxRadius=100
)

# If circles are found, draw bounding boxes around them
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # Get the bounding box coordinates
        x, y, r = i
        x, y, w, h = x - r, y - r, 2*r, 2*r

        # Add conditions to filter out unwanted circles based on LBP features (adjust these conditions as needed)
        lbp_roi = lbp_result[y:y+h, x:x+w]
        unique_labels, counts = np.unique(lbp_roi, return_counts=True)

        # Example condition: consider circles with more than 80% uniform LBP patterns
        if counts[0] / np.sum(counts) < 0.8:
            # Draw the bounding box in green
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            roi = thresh[y:y + h, x:x + w]
            roismall = cv2.resize(roi, (35, 35))
            # cv2.imshow('norm', roi)
            
key = cv2.waitKey(0)
# Save the image with potential prohibition signs highlighted
output_path_prohibition_signs = 'hello.png'
cv2.imwrite(output_path_prohibition_signs, image)
