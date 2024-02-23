import cv2
import numpy as np
from keras.models import load_model

model = load_model('model.h5')

prohibition_classes = {
                        1: 'Speed limit (20km/h)',
                        2: 'Speed limit (30km/h)',
                        3: 'Speed limit (50km/h)',
                        4: 'Speed limit (60km/h)',
                        5: 'Speed limit (70km/h)',
                        6: 'Speed limit (80km/h)',
                        7: 'No car',
                        8: 'No car > 2.5',
                        9: 'No bicycle',
                        10: 'No passing',
                        11: 'No passing veh over 3.5 tons',
                        12: 'No vehicles',
                        13: 'Veh > 3.5 tons prohibited',
                        14: 'No entry',
                        15: 'Beware of ice/snow',
                        16: 'End of no passing',
                        17: 'No Turn Left',
                        18: 'No Turn Right',
                        19: 'Stop'}

#####################################################
# Load the original image
image_path = r'input1.png'
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
    maxRadius=70
)

color_index = 0
colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        x, y, r = i
        x, y, w, h = x - r, y - r, 2*r, 2*r
        lbp_roi = lbp_result[y:y+h, x:x+w]
        unique_labels, counts = np.unique(lbp_roi, return_counts=True)

        current_color = colors[color_index]
        color_index = (color_index + 1) % len(colors)

        if counts[0] / np.sum(counts) < 0.8:
            cv2.rectangle(image, (x, y), (x+w, y+h), current_color, 2)

        roi_color = image[y:y+h, x:x+w]
        roi_color_rgb = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
        roi_color_resized = cv2.resize(roi_color_rgb, (30, 30))  
        roi_color_resized = roi_color_resized.astype('float32') / 255.0 
        image_array = np.expand_dims(roi_color_resized, axis=0)

        # Dự đoán
        pred_probabilities = model.predict(image_array)
        pred_class = np.argmax(pred_probabilities, axis=1)[0]
        sign = prohibition_classes[pred_class + 1]

        # Ghi chú tên biển báo lên ảnh
        cv2.putText(image, sign, (x, y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, current_color, 2)

output_path_prohibition_signs = 'output.png'
cv2.imwrite(output_path_prohibition_signs, image)