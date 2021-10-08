"""A machine learning to support operations via roboflow.ai
"""
import requests
import base64
import io
import os
import json

_CONFIG_FILE = "config.json"

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
            self.modelUuid = config['modelUuid']
            self.roboflowKey = config['roboflowKey']
            self.uploadNewImages = config['uploadNewImages']
       
    def upload_image(self, image: Image.Image,
                     imageName: str):
        """Code to upload an image to Roboflow.
           Code is borrowed from the example at:
           https://docs.roboflow.com/adding-data/upload-api
           :param image: Input image that needs to be uploaded
        """
        #First check if permissions are available to upload new images.
        if not self.uploadNewImages:
            return
        # Convert to JPEG Buffer
        buffered = io.BytesIO()
        image.save(buffered, quality=90, format="JPEG")

        # Base 64 Encode
        img_str = base64.b64encode(buffered.getvalue())
        img_str = img_str.decode("ascii")

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
        result = requests.post(upload_url, data=img_str, headers={
            "Content-Type": "application/x-www-form-urlencoded"
            })

        res = result.json()
        success = res.get('success')
        if not success:
            print(res)
        else:
            print("Image %s uploaded successfully!" % imageName)

    def run_inference(self, image: Image.Image): 
        """Run inference on the provided image with the help of Roboflow
        inference API. Returns an annotated Image in case inference detects
        an object
        :param image: The image to run inference on
        :return Returns a tuple of an image with bounding boxes around the
        objects detected and a set of tags of all the objects detected
        """
        # Convert to JPEG Buffer
        buffered = io.BytesIO()
        image.save(buffered, quality=90, format="JPEG")

        # Base 64 Encode
        img_str = base64.b64encode(buffered.getvalue())
        img_str = img_str.decode("ascii")

        # Construct the Roboflow URL to do Inference
        upload_url = "".join([
            "https://detect.roboflow.com/",
            self.dataset,
            "/",
            self.modelUuid,
            "?api_key=",
            self.roboflowKey, 
            "&format=json" 
        ])

        # POST request to the API
        headers = {'accept': 'application/json'}
        r = requests.post(upload_url, data=img_str, headers=headers)
        preds = r.json()
        detections = preds['predictions']

        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        for box in detections:
            color = "#4892EA"
            x1 = box['x'] - box['width'] / 2
            x2 = box['x'] + box['width'] / 2
            y1 = box['y'] - box['height'] / 2
            y2 = box['y'] + box['height'] / 2
            draw.rectangle([
                x1, y1, x2, y2
            ], outline=color, width=5)

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
