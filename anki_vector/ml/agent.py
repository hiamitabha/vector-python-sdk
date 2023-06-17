"""A machine learning to support operations via roboflow.ai
"""
import requests
import io
import os
import json
from requests_toolbelt.multipart.encoder import MultipartEncoder

_CONFIG_FILE = "config.json"
_MODEL_SWAP_ITERATIONS = 20
_VERSION_COLOR_CODE = '#8e3c44'

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    sys.exit("Cannot import from PIL: Do `pip3 install --user Pillow` to install")

class MLAgent:
    """MLAgent drives all ML driven operations with the help of support provided by roboflow.ai
    """
    def __init__(self):
        current_dir = os.path.dirname(os.path.realpath(__file__)) 
        config_file = os.path.join(current_dir,_CONFIG_FILE)
        with open(config_file) as json_file:
            config = json.load(json_file)
            self.dataset = config['dataset']
            self.modelUuid = config['modelUuid'].split(",")
            self.roboflowKey = config['roboflowKey']
            self.uploadNewImages = config['uploadNewImages']
            self.type = config.get('type')
            self.numModels = len(self.modelUuid)
            self.currentModelId = 0
            self.modelUseCounter = 0
       
    def upload_image(self, image: Image.Image,
                     imageName: str):
        """Code to upload an image to Roboflow.
           Code is borrowed from the example at:
           https://docs.roboflow.com/adding-data/upload-api
           :param image: Input image that needs to be uploaded
           :param imageName: The name with which the image needs to be uploaded
        """
        #First check if permissions are available to upload new images.
        if not self.uploadNewImages:
            return
        # Convert to JPEG Buffer
        buffered = io.BytesIO()
        image.save(buffered, quality=90, format="JPEG")

        # Build multipart form and post request
        m = MultipartEncoder(fields={'file': ("imageToUpload", buffered.getvalue(), "image/jpeg")})

        # Construct the URL
        upload_url = "".join([
            "https://api.roboflow.com/dataset/",
            self.dataset,
            "/upload",
            "?api_key=",
            self.roboflowKey,
            "&name=",
            imageName,
            "&split=train"
            ])

        # POST to the API
        result = requests.post(upload_url, data=m, headers={
            "Content-Type": m.content_type 
            })

        res = result.json()
        success = res.get('success')
        if not success:
            print(res)
        else:
            print("Image %s uploaded successfully!" % imageName)

    def updateCurrentModelId(self):
        """Updates the model Id depending on the usage to facilitate A/B testing
        """
        self.modelUseCounter += 1
        if (self.modelUseCounter % _MODEL_SWAP_ITERATIONS == 0):
            self.currentModelId +=1
            if (self.currentModelId == self.numModels):
                self.currentModelId = 0

    def run_inference_via_roboflow(self, image: Image.Image):
        """Run inference via roboflow APIs. Returns image with bounded boxes
        :param image: The image to run inference on
        :return Returns a tuple of an image with bounding boxes around the
        objects detected and a set of tags of all the objects detected
        """
        # Convert to JPEG Buffer
        buffered = io.BytesIO()
        image.save(buffered, quality=90, format="JPEG")

        # Build multipart form and post request
        m = MultipartEncoder(fields={'file': ("imageToUpload", buffered.getvalue(), "image/jpeg")})
        self.updateCurrentModelId()
        model = self.modelUuid[self.currentModelId] 
        # Construct the Roboflow URL to do Inference
        if self.type == 'object-detection':
            upload_url = "".join([
                "https://detect.roboflow.com/",
                self.dataset,
                "/",
                model,
                "?api_key=",
                self.roboflowKey,
                "&format=json"
            ])
        elif self.type == 'instance-segmentation'
            upload_url = "".join([
                "https://outline.roboflow.com/",
                self.dataset,
                "/",
                model,
                "?api_key=",
                self.roboflowKey,
                "&format=json"
            ])

        # POST request to the API
        r = requests.post(upload_url, data=m, headers={
            "Content-Type": m.content_type
            })
        preds = r.json()
        detections = preds['predictions']

        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        draw.text((10,10), "Running model %s version %s " % (self.dataset, model),
                  fill=_VERSION_COLOR_CODE)
       
        for box in detections:
            color = "#4892EA"
            x1 = box['x'] - box['width'] / 2
            y1 = box['y'] - box['height'] / 2
            if self.type == 'object-detection':
                x2 = box['x'] + box['width'] / 2
                y2 = box['y'] + box['height'] / 2
                draw.rectangle([
                    x1, y1, x2, y2
                ], outline=color, width=5)
            elif self.type == 'instance-segmentation':
                points = box.get('points')
                start_x = points[0]['x']
                start_y = points[0]['y']
                for point in points[1:]:
                    next_x = point['x']
                    next_y = point['y']
                    draw.line([start_x, start_y, next_x, next_y], fill=color, width=5)
                    start_x = next_x
                    start_y = next_y

            #text = box['class'] + '_modelv_' + model
            text = box['class']
            text_size = font.getsize(text)

            #set button size + 10px margins
            button_size = (text_size[0]+20, text_size[1]+20)
            button_img = Image.new('RGBA', button_size, color)
            # put text on button with 10px margins
            button_draw = ImageDraw.Draw(button_img)
            button_draw.text((10, 10), text, font=font, fill=(255,255,255,255))

            # put button on source image in position (0, 0)
            image.paste(button_img, (int(x1), int(y1)))
        return image

    def run_inference_via_custom_url(self, image: Image.Image):
        """Run inference via a custom URL.
        Future: Read the custom URL from a config file.
        """
        buffered = io.BytesIO()
        image.save(buffered, quality=90, format="JPEG")
        url = 'http://localhost:5000/'
        files = {'file': buffered.getvalue(), 'model_choice':'best_s'}
        response = requests.post(url, files=files)
        image = Image.open(io.BytesIO(response.content))
        return image

    def run_inference(self, image: Image.Image): 
        """Run inference on the provided image with the help of Roboflow
        inference API. Returns an annotated Image in case inference detects
        an object
        :param image: The image to run inference on
        :return Returns a tuple of an image with bounding boxes around the
        objects detected and a set of tags of all the objects detected
        """
        image = self.run_inference_via_roboflow(image)
        return image
