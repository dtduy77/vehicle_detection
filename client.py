import math
from PIL import Image, ImageDraw
import cv2
import numpy as np
import tritonclient.http as httpclient
from main import *
import tritonclient.grpc as grpcclient

class CFG:
    image_size = IMAGE_SIZE
    conf_thres = 0.01
    iou_thres = 0.1

cfg = CFG()

if __name__ == "__main__":
    # Setting up client
    client = grpcclient.InferenceServerClient(url="localhost:8001")

    # Open a file dialog to select an image
    image_path = "1.jpg"
    raw_image = cv2.imread(image_path)
    # resized_pad_image, ratio, (padd_left, padd_top) = resize_and_pad(raw_image, new_shape=cfg.image_size)
    # norm_image = normalization_input(resized_pad_image)

    detection_input = grpcclient.InferInput("images", raw_image.shape, datatype="FP32")
    detection_input[0].set_data_from_numpy(raw_image)
    detection_response = client.infer(model_name="ensemble_model", inputs=detection_input)
    pred = detection_response.as_numpy("output0")
    # pred = postprocess(result, cfg.conf_thres, cfg.iou_thres)[0]

    paddings = np.array([padd_left, padd_top, padd_left, padd_top])
    pred[:,:4] = (pred[:,:4] - paddings) / ratio

    # Visualize the image with bounding box predictions
    visualize_image(raw_image, pred)

