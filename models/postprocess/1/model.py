import numpy as np
import json
import triton_python_backend_utils as pb_utils
import cv2
from PIL import Image

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])

        # Get OUTPUT0 configuration
        postprocess_config = pb_utils.get_output_config_by_name(
            model_config, "postprocess_output")
        

        # Convert Triton types to numpy types
        self.postprocess_dtype = pb_utils.triton_string_to_numpy(
            postprocess_config['data_type'])

    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        postprocess_dtype = self.postprocess_dtype
        
        responses = []

        def convert_xywh_to_xyxy(bbox_array: np.array) -> np.array:
            converted_boxes = np.zeros_like(bbox_array)
            converted_boxes[:, 0] = bbox_array[:, 0] - bbox_array[:, 2] / 2  # x1 (top-left x)
            converted_boxes[:, 1] = bbox_array[:, 1] - bbox_array[:, 3] / 2  # y1 (top-left y)
            converted_boxes[:, 2] = bbox_array[:, 0] + bbox_array[:, 2] / 2  # x2 (bottom-right x)
            converted_boxes[:, 3] = bbox_array[:, 1] + bbox_array[:, 3] / 2  # y2 (bottom-right y)

            return converted_boxes


        def calculate_iou(box1: np.array, box2: np.array) -> float:
            x1_1, y1_1, x2_1, y2_1 = box1
            x1_2, y1_2, x2_2, y2_2 = box2
            # Calculate the coordinates of the intersection rectangle
            x1_i = max(x1_1, x1_2)
            y1_i = max(y1_1, y1_2)
            x2_i = min(x2_1, x2_2)
            y2_i = min(y2_1, y2_2)

            # Calculate the area of intersection rectangle
            intersection_area = max(0, x2_i - x1_i + 1) * max(0, y2_i - y1_i + 1)

            # Calculate the area of both input rectangles
            area1 = (x2_1 - x1_1 + 1) * (y2_1 - y1_1 + 1)
            area2 = (x2_2 - x1_2 + 1) * (y2_2 - y1_2 + 1)

            # Calculate IoU
            iou = intersection_area / float(area1 + area2 - intersection_area)

            return iou

        def nms(bboxes: np.array, scores: np.array, iou_threshold: float) -> np.array:
            selected_indices = []

            # Sort bounding boxes by decreasing confidence scores
            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

            while len(sorted_indices) > 0:
                current_index = sorted_indices[0]
                selected_indices.append(current_index)

                # Remove the current box from the sorted list
                sorted_indices.pop(0)

                indices_to_remove = []
                for index in sorted_indices:
                    iou = calculate_iou(bboxes[current_index], bboxes[index])
                    if iou >= iou_threshold:
                        indices_to_remove.append(index)

                # Remove overlapping boxes from the sorted list
                sorted_indices = [i for i in sorted_indices if i not in indices_to_remove]

            return selected_indices

        def postprocess(prediction: np.array, conf_thres: float=0.15, iou_thres: float=0.45, max_det: int=300) -> np.array:
            bs = prediction.shape[0]  # batch size
            xc = prediction[..., 4] > conf_thres  # candidates
            max_nms = 300  # maximum number of boxes into NMS
            max_wh = 7680
            output = [None] * bs

            for xi, x in enumerate(prediction):
                x = x[xc[xi]]
                if len(x) == 0:
                    continue
                x[:, 5:] *= x[:, 4:5]
                # Define xywh2xyxy_numpy function or import it
                box = convert_xywh_to_xyxy(x[:, :4])

                # Detections matrix nx6 (xyxy, conf, cls)
                conf = x[:, 5:].max(1)
                max_conf_indices = x[:, 5:].argmax(1)
                x = np.column_stack((box, conf, max_conf_indices.astype(float)))[conf > conf_thres]

                n = len(x)
                if n == 0:
                    continue
                elif n > max_nms:
                    sorted_indices = np.argsort(-x[:, 4])
                    x = x[sorted_indices[:max_nms]]

                # Batched NMS
                c = x[:, 5:6] * max_wh  # You should compute max_wh based on image dimensions
                boxes, scores = x[:, :4] + c, x[:, 4]
                # Define nms_boxes_numpy function or import it
                i = nms(boxes, scores, iou_thres)
                if len(i) > max_det:
                    i = i[:max_det]
                output[xi] = x[i]
            return output

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get image
            postprocess = pb_utils.get_input_tensor_by_name(request, "output0")
            pred = postprocess(pred)[0]

            out_postprocess = pb_utils.Tensor("postprocess_output", pred.astype(postprocess_dtype))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_postprocess])
            responses.append(inference_response)

        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        pass

