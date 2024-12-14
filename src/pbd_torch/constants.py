import torch
import numpy as np

from pbd_torch.transform import rotvec_to_quat

# Identity quaternion
ROT_IDENTITY = torch.tensor([1.0, 0.0, 0.0, 0.0])

# 45-degree rotations for each axis
ROT_45_X = rotvec_to_quat(torch.tensor([np.pi / 4, 0.0, 0.0]))
ROT_45_Y = rotvec_to_quat(torch.tensor([0.0, np.pi / 4, 0.0]))
ROT_45_Z = rotvec_to_quat(torch.tensor([0.0, 0.0, np.pi / 4]))

# Negative 45-degree rotations
ROT_NEG_45_X = rotvec_to_quat(torch.tensor([-np.pi / 4, 0.0, 0.0]))
ROT_NEG_45_Y = rotvec_to_quat(torch.tensor([0.0, -np.pi / 4, 0.0]))
ROT_NEG_45_Z = rotvec_to_quat(torch.tensor([0.0, 0.0, -np.pi / 4]))

# 90-degree rotations for each axis
ROT_90_X = rotvec_to_quat(torch.tensor([np.pi / 2, 0.0, 0.0]))
ROT_90_Y = rotvec_to_quat(torch.tensor([0.0, np.pi / 2, 0.0]))
ROT_90_Z = rotvec_to_quat(torch.tensor([0.0, 0.0, np.pi / 2]))

# Negative 90-degree rotations
ROT_NEG_90_X = rotvec_to_quat(torch.tensor([-np.pi / 2, 0.0, 0.0]))
ROT_NEG_90_Y = rotvec_to_quat(torch.tensor([0.0, -np.pi / 2, 0.0]))
ROT_NEG_90_Z = rotvec_to_quat(torch.tensor([0.0, 0.0, -np.pi / 2]))

# Half rotations (180 degrees) for each axis
ROT_180_X = rotvec_to_quat(torch.tensor([np.pi, 0.0, 0.0]))
ROT_180_Y = rotvec_to_quat(torch.tensor([0.0, np.pi, 0.0]))
ROT_180_Z = rotvec_to_quat(torch.tensor([0.0, 0.0, np.pi]))

# Combined rotations 45° about two axes
ROT_45_XY = rotvec_to_quat(torch.tensor([np.pi / 4, np.pi / 4, 0.0]))
ROT_45_XZ = rotvec_to_quat(torch.tensor([np.pi / 4, 0.0, np.pi / 4]))
ROT_45_YZ = rotvec_to_quat(torch.tensor([0.0, np.pi / 4, np.pi / 4]))

# Combined rotations 90° about two axes
ROT_90_XY = rotvec_to_quat(torch.tensor([np.pi / 2, np.pi / 2, 0.0]))
ROT_90_XZ = rotvec_to_quat(torch.tensor([np.pi / 2, 0.0, np.pi / 2]))
ROT_90_YZ = rotvec_to_quat(torch.tensor([0.0, np.pi / 2, np.pi / 2]))

# Matplotlib colors
BLUE = (0.12156862745098039, 0.4666666666666667, 0.7058823529411765)
ORANGE = (1.0, 0.4980392156862745, 0.054901960784313725)
GREEN = (0.17254901960784313, 0.6274509803921569, 0.17254901960784313)
RED = (0.8392156862745098, 0.15294117647058825, 0.1568627450980392)
PURPLE = (0.5803921568627451, 0.403921568627451, 0.7411764705882353)
BROWN = (0.5490196078431373, 0.33725490196078434, 0.29411764705882354)
PINK = (0.8901960784313725, 0.4666666666666667, 0.7607843137254902)
GRAY = (0.4980392156862745, 0.4980392156862745, 0.4980392156862745)
OLIVE = (0.7372549019607844, 0.7411764705882353, 0.13333333333333333)
CYAN = (0.09019607843137255, 0.7450980392156863, 0.8117647058823529)
