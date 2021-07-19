import argparse
import random
import cv2
import darknet

'''
Purpose:
Detect logo from an image, and draw detected result (bounding boxes, label name) on the image
'''

'''
TODO:
- support multiple input format
'''


def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default="7-1.PNG",
                        help="image source. It can be a single image, a"
                        "txt with paths to them, or a folder. Image valid"
                        " formats are jpg, jpeg or png."
                        "If no input is given, ")
    parser.add_argument("--batch_size", default=1, type=int,
                        help="number of images to be processed at the same time")
    parser.add_argument("--weights", default="backup/yolov4-custom_best.weights",
                        help="yolo weights path")
    parser.add_argument("--config_file", default="cfg/yolov4-custom.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="data/obj.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.9,
                        help="remove detections with lower confidence")
    parser.add_argument("--font_path", type=str, default="font/edukai-3.ttf",
                        help="remove detections with lower confidence")
    return parser.parse_args()


def main():
    args = parser()
    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=args.batch_size
    )

    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    image = cv2.imread(args.input)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=args.thresh)
    darknet.free_image(darknet_image)

    image = darknet.draw_boxes_chinese(detections, image_resized, class_colors, args.font_path)
    # return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections
    cv2.imwrite('result.jpg', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))




if __name__ == "__main__":
    main()