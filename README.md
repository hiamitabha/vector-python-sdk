# Amitabha's Anki Vector - Python SDK

## Amitabha's modifications to Official Anki Vector SDK

Current (v 0.7.0.hiamitabha):
- Ability to detect and view another Vector robot with the help of the viewer

Future/Planned:
- Raising an event when a vector robot appears/ dissappears, is first observed.
- Ability to track and distinguish multiple vectors with the color of their eye

## Notes on usage
This SDK uses an ML inference API to detect other Vector robots. The API service is
provided by Roboflow. You have two ways to use this:

1) You could use the Roboflow API with my account credentials. In that case you would have to
fill in the parameters specified in ml/config.json with those provided by me. To get these
parameters, you would need to message me via contacts listed on my github profile:
https://github.com/hiamitabha Please adhere to the restrictions described in my email, as any
abuse of the services provided by my account effects everybody using the service.

2) You could fork my public dataset at https://public.roboflow.com/object-detection/robot
After forking, you can generate your own dataset by choosing various preprocessing and augmentation
options that Roboflow provides. Thereafter, you would have to train your model at Roboflow, so that
they can provide you with an API to do inference. You can then pick up the datset name, model version,
and roboflow key from the curl URL and specify them in ml/config.json. Complete details on how to
train a model at Roboflow is available at https://docs.roboflow.com/train

## Execution
To get a feeling of the power of this SDK, please try the following program:

python3 examples/tutorials/19_show_video_feed.py

Here is an example output 
[video](https://youtu.be/Nw9a50zGnvs)
from this program.

## Feedback
I have invested considerable time and effort building this SDK, and any feedback that you can provide
would be very helpful. Please reach me via my contacts at my github profile: https://github.com/hiamitabha
for any feedback


# Generic Notes from standard version of Vector SDK (v0.6)

![Vector](docs/source/images/vector-sdk-alpha.jpg)

Learn more about Vector: https://www.anki.com/en-us/vector

Learn more about the SDK: https://developer.anki.com/

SDK documentation: https://developer.anki.com/vector/docs/index.html

Forums: https://forums.anki.com/


## Getting Started

You can follow steps [here](https://developer.anki.com/vector/docs/index.html) to set up your Vector robot with the SDK.

You can also generate a local copy of the SDK documetation by
following the instructions in the `docs` folder of this project.


## Privacy Policy and Terms and Conditions

Use of Vector and the Vector SDK is subject to Anki's [Privacy Policy](https://www.anki.com/en-us/company/privacy) and [Terms and Conditions](https://www.anki.com/en-us/company/terms-and-conditions).
