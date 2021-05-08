import torch
from torch import nn

# Deformable Convolutional Networks
# https://arxiv.org/abs/1703.06211
"""
y(p_0) = \sum_{p_n \in R}w(p_n)x(p_0 + p_n + \delta p_n)
"""
class DeformableConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, padding = 1,
            stride = 1, bias = None):
        super(DeformableConvolution, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv_kernel = nn.Conv2d(in_channels, out_channels,
                kernel_size = kernel_size, stride = kernel_size, bias = bias)
        self.offsets = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size,
                kernel_size = 3, padding = 1, stride = stride)
        self.offsets.register_backward_hook(self.set_lr)

    def set_lr(module, input, output):
        input = (input[i] * 0.1 for i in range(len(input)))
        output = (output[i] * 0.1 for i in range(len(output)))

    def forward(self, x):
        # R = {(x_1, y_2),..., (x_n, y_n)} | n = 1,..., N
        offset = self.offsets(x)
        dtype = offset.data.type()
        kernel_size = self.kernel_size
        # N = |R|
        N = offset.size(1) // 2
        if self.padding:
            x = self.zero_padding(x)
        # (b, 2N, h, w)
        p = self.get_p(offset, dtype)
        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1),
            torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)], dim = -1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1),
            torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)], dim = -1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim = -1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim = -1)
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1),
            torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim = -1)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * \
                (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * \
                (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * \
                (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * \
                (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))
        x_q_lt = self.get_x_q(x, q_lt, N)
        x_q_rb = self.get_x_q(x, q_rb, N)
        x_q_lb = self.get_x_q(x, q_lb, N)
        x_q_rt = self.get_x_q(x, q_rt, N)
        # x(p) = \sum_q G(q, p) * x(q)
        # where G(q, p) = g(q_x, p_x) * g(q_y, q_x)
        x_offset = g_lt.unsqueeze(dim = 1) * x_q_lt + g_rb.unsqueeze(dim = 1
                ) * x_q_rb + g_lb.unsqueeze(dim = 1) * x_q_lb + g_rt.unsqueeze(
                        dim = 1) * x_q_rt
        x_offset = self.reshape_x_offset(x_offset, kernel_size)
        return self.conv_kernel(x_offset)
    
    def get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(torch.arange(1, h * self.stride + 1,
            self.stride), torch.arange(1, w *  self.stride + 1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        return p_0

    def get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
                torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1
                    ) // 2 + 1), torch.arange(-(self.kernel_size - 1) // 2, (
                        self.kernel_size - 1) // 2 + 1))
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    def get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)
        p_0 = self.get_p_0(h, w, N, dtype)
        p_n = self.get_p_n(N, dtype)
        p = p_0 + p_n + offset
        return p

    def get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        x = x.contiguous().view(b, c, -1)
        index = q[..., :N] * padded_w + q[..., N:]
        # (b, c, h * w * N)
        index = index.contiguous().unsqueeze(dim = 1).expand(-1, c, -1, -1, -1
                ).contiguous().view(b, c, -1)
        x_offset = x.gather(dim = -1, index = index).contiguous().view(b, c, h,
                w, N)
        return x_offset

    def reshape_x_offset(self, x_offset, kernel_size):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s: s + kernel_size].contiguous().
            view(b, c, h, w * kernel_size) for s in range(0, N, kernel_size)],
            dim = -1).contiguous().view(b, c, h * kernel_size, w * kernel_size)
        return x_offset
