import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

class YOLOv8Node(Node):
    def __init__(self):
        super().__init__('yolov8_node')
        self.bridge = CvBridge()
        self.model = YOLO("yolov8n-seg.pt")
        self.pub = self.create_publisher(Image, 'yolov8_detections', 10)
        self.cap = cv2.VideoCapture(0)  # Use laptop webcam (index 0)
        self.timer = self.create_timer(0.1, self.detect_and_publish)

    def detect_and_publish(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        results = self.model(frame, show_boxes=False)
        annotated_frame = results[0].plot()

        # Convert the annotated frame to ROS Image message
        img_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
        self.pub.publish(img_msg)

def main(args=None):
    rclpy.init(args=args)
    node = YOLOv8Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()