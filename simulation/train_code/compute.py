from thop import profile
from architecture import *
import torch
import os
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    # input_tensor1 = torch.randn(256, 256)  # 不需要.cuda()，此处仅生成输入张量，不进行计算
    RGB = torch.randn(1, 256, 310).cuda() # 将输入张量移至 GPU
    input_mask=torch.randn(1, 28, 256, 310).cuda() ,torch.randn(1,256, 310).cuda()
    model = MIDET(28,28,8).to('cuda:0') # 将模型移至 GPU
    # model = DMDC(1,31)  # 将模型移至 GPU
    with torch.no_grad():
        # macs, params = profile(model, inputs=(input_tensor1,RGB,input_mask))
        macs, params = profile(model, inputs=(RGB,input_mask))
        # 使用(,)确保传递的是单个元素的 tuple

    print('Parameters number is ', sum(param.numel() for param in model.parameters()))
    print('Parameters number is {}; Flops: {}'.format(params, macs))
    print(torch.__version__)