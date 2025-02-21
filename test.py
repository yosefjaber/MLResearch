import torch
import torch.nn as nn
import torch.nn.functional as F
from evaluate import eval_model
from tarikmodel3 import tarikModel3
from train_model import train_model
from create_model import create_model
import gc

#5 layers
# small_left_pyramid = create_model(4,[8,16,32,64,128],1)
# small_right_pyramid = create_model(4,[128,64,32,16,8],1)
# small_diamond = create_model(4,[8,16,32,16,4],1)
# small_block = create_model(4,[16384,16384,16384,16384,16384],1)

#10 layers
# medium_left_pyramid = create_model(4,[8,16,32,64,128,256,512,1024,2048,4096],1)
#medium_right_pyramid = create_model(4,[4096,2048,1024,512,256,128,64,32,16,8],1)
# medium_diamond = create_model(4,[8,16,32,64,128,256,64,32,16,8],1)
# medium_block = create_model(4,[16384,16384,16384,16384,16384,16384,16384,16384,16384,16384],1)

#14 layers
#
# large_left_pyramid = create_model(4,[8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536],1)

# test_model(large_left_pyramid, 0.001, "AdamW", 600, "testWeights.pt")
# eval_model("testWeights.pt", False)
# del large_left_pyramid
# gc.collect()
# torch.cuda.empty_cache()

# large_left_pyramid = create_model(4,[8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536],1)
# large_right_pyramid = create_model(4,[65536,32768,16384,8192,4096,2048,1024,512,256,128,64,32,16,8],1)
# large_diamond = create_model(4,[8,16,32,64,128,256,512,512,256,128,64,32,16,8],1)
#large_block = create_model(4,[16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384,16384],1)
#massive_block = create_model(4,[4096,4096,4096,4096,4096,4096,4096,4096,4096,4096,4096,4096,4096,4096,4096,4096],1)


# test_model(large_block, 0.001, "AdamW", 600, "testWeights.pt")
# eval_model("testWeights.pt", False)


test_blockRAM = create_model(4,[1024 for i in range(18)],1)

#train_model(test_blockRAM, 0.0001, "AdamW", 450_000, "test_blockRAM.pt") #Use 600_000 lr = 0.000004 2.1
train_model(test_blockRAM, 0.0000001, "AdamW", 550_000, "test_blockRAM.pt")
eval_model(test_blockRAM,"test_blockRAM.pt", True)
torch.cuda.empty_cache()