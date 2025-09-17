# 此文件修复 No module named 'torchvision.transforms.functional_tensor'
#
# 问题背景:
# 在较新版本的torchvision(0.17+)中，torchvision.transforms.functional_tensor模块已被移除或合并到
# torchvision.transforms.functional中，但某些依赖项或旧代码仍在尝试导入该模块，导致出现ImportError错误。
#
# 解决方案:
# 通过sys.modules将torchvision.transforms.functional_tensor指向torchvision.transforms.functional，
# 实现模块重定向，解决导入错误问题。
import torchvision.transforms.functional as functional
import sys
# 将torchvision.transforms.functional_tensor 模块重定向到 torchvision.transforms.functional
sys.modules['torchvision.transforms.functional_tensor'] = functional