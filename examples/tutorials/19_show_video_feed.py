#!/usr/bin/env python3

# Copyright (c) 2021 Amitabha Banerjee
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License in the file LICENSE.txt or at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''Display a GUI window with video feed from Vector
'''

import time
import argparse

import anki_vector
from anki_vector.util import degrees

if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   args = anki_vector.util.parse_command_args(parser)
   with anki_vector.Robot(serial=args.serial, show_viewer=True) as robot:
      robot.behavior.set_head_angle(degrees(3.0))
      time.sleep(240)
      robot.viewer.close()
