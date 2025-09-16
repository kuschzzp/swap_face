
# 此文件修复 [ROOP.CORE] Processing failed with error: No module named 'torchvision.transforms.functional_tensor' 

import torchvision.transforms.functional as functional
import sys
sys.modules['torchvision.transforms.functional_tensor'] = functional