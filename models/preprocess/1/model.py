import numpy as np
import json
import triton_python_backend_utils as pb_utils
import cv2

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
        preprocess_config = pb_utils.get_output_config_by_name(
            model_config, "preprocess_output")
        

        # Convert Triton types to numpy types
        self.preprocess_dtype = pb_utils.triton_string_to_numpy(
            preprocess_config['data_type'])

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

        preprocess_dtype = self.preprocess_dtype
        
        responses = []

        def resize_and_pad(image: np.array, 
                   new_shape: Tuple[int, int], 
                   padding_color: Tuple[int] = (144, 144, 144)
                   ) -> np.array:
            h_org, w_org = image.shape[:2]
            w_new, h_new = new_shape
            padd_left, padd_right, padd_top, padd_bottom = 0, 0, 0, 0

            #Padding left to right
            if h_org >= w_org:
                img_resize = cv2.resize(image, (int(w_org*h_new/h_org), h_new))
                h, w = img_resize.shape[:2]
                padd_left = (w_new-w)//2
                padd_right =  w_new - w - padd_left
                ratio = h_new/h_org

            #Padding top to bottom
            if h_org < w_org:
                img_resize = cv2.resize(image, (w_new, int(h_org*w_new/w_org)))
                h, w = img_resize.shape[:2]
                padd_top = (h_new-h)//2
                padd_bottom =  h_new - h - padd_top
                ratio = w_new/w_org
            
            image = cv2.copyMakeBorder(img_resize, padd_top, padd_bottom, padd_left, padd_right, cv2.BORDER_CONSTANT,None,value=padding_color)
            
            return image, ratio, (padd_left, padd_top)

        def normalization_input(image:  np.array) ->  np.array:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #BGR to RGB
            img = image.transpose((2, 0, 1)) # HWC to CHW
            img = np.ascontiguousarray(img).astype(np.float32)
            img /=255.0
            img = img[np.newaxis, ...]
            return img

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get image
            image_cv = pb_utils.get_input_tensor_by_name(request, "preprocess_input")
            image, ratio, (padd_left, padd_top) = resize_and_pad(image_cv, new_shape=(448,448)) #(448,448)
            img_norm = normalization_input(image)

            out_preprocess = pb_utils.Tensor("preprocess_output", img_norm.astype(preprocess_dtype))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_preprocess])
            responses.append(inference_response)

        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        pass
