with open("utils/postprocess.py", "r") as f:
    text = f.read()

import re

target = """        d.addPairwiseGaussian(sxy=3, compat=7)
        d.addPairwiseBilateral(
            sxy=50, srgb=5,
            rgbim=np.ascontiguousarray(np.array(original_image), dtype=np.uint8),
            compat=10,
        )"""

replacement = """        d.addPairwiseGaussian(sxy=3, compat=7)
        d.addPairwiseBilateral(
            sxy=50, srgb=5,
            rgbim=np.ascontiguousarray(np.array(original_image), dtype=np.uint8),
            compat=10,
        )"""

# Actually they were exactly identical already. I checked before.
