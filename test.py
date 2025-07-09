import cv2

# Load an image from the specified path
image_path = '/home/xjc/dataset/kitti_tracking/training/image_02/0000/000014.png'  # Replace with your image path
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print(f"Error: Unable to load image at {image_path}")
else:
    # Display the image in a window
    cv2.imshow('Image Viewer', image)
    
    # Wait for a key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()