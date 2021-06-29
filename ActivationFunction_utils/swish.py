import torch

class SwishImplementation(torch.autograd.Function):
    """
    ctx为自动生成的一个反向传播相关类对象,这个对象保存了处理流程中所需的数据
    i即为模块的输入张量
    """
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result
    # grad_output即为上层传入的导数值,可以根据自定义的导函数进行反向传播
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


if __name__ == '__main__':
    swish = SwishImplementation()
    input = torch.randn(2,2,requires_grad=True)
    y = swish.apply(input)
    y.mean().backward()