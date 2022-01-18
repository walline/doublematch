#!/usr/bin/env python3

# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import sys
import numpy as np

assert len(sys.argv[1:]) == 5

vals_dict = {"max": [],
             "last20": []}

for arg in sys.argv[1:]:
    with open(arg, 'r') as f:
        d = json.load(f)

    vals_dict["max"].append(d["max"])
    vals_dict["last20"].append(d["last20"])


for key, vals in vals_dict.items():
    mean = 100.0 - np.mean(vals) # mean error rate
    std = np.std(vals)

    print("{}: {:.2f} +- {:.2f}".format(key, mean, std))
