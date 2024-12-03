import numpy as np
import torch


def conv2d_cpu_torch(X, W, bias, pad_size=0, pool_size=2):
    X = torch.tensor(X)
    W = torch.tensor(W)
    bias = torch.tensor(bias)

    conv_out = torch.nn.functional.conv2d(X, W, bias, stride=1, padding=pad_size)

    if pool_size > 1:
        return torch.nn.functional.max_pool2d(
            conv_out, kernel_size=pool_size, stride=pool_size
        )

    return conv_out

def conv2d_myref(X, F, bias, pad_size=0, pool_size=2):
    B, C, H, W = X.shape
    O, C_, K, K_ = F.shape
    O_ = bias.shape[0]

    assert(C == C_)
    assert(K == K_)
    assert(O == O_)
    assert(C % 128 == 0)
    assert(O % 128 == 0)

    H_out = H - K + 1
    W_out = W - K + 1
    H_pool = H_out // pool_size
    W_pool = W_out // pool_size

    C_tile = 128
    n_Ctiles = C // C_tile

    O_tile = 128
    n_Otiles = O // O_tile

    n_rowpairs = H_out // 2

    print("hiii\n")
    print(B, C, H, W)
    print(O, C, K)
    print(H_out, W_out)
    print(pool_size, H_pool, W_pool)
    print(C_tile, n_Ctiles)
    print(O_tile, n_Otiles)
    print(n_rowpairs)

    X_out = np.zeros((B, O, H_pool, W_pool))

    input_tiles = np.zeros((K+1, n_Ctiles, C_tile, W))
    kernel_tiles = np.zeros((K, K, n_Ctiles, C_tile, O_tile))
    bias_tile = np.zeros((O_tile, 1))
    out_tiles = np.zeros((2, O_tile, W_out))

    # For each image
    for b in range(B):
        # For each group of kernels
        for o in range(n_Otiles):
            o_start = o*O_tile
            o_end = (o+1)*O_tile
            # Load kernel
            for ki in range(K):
                for kj in range(K):
                    for c in range(n_Ctiles):
                        c_start = c*C_tile
                        c_end = (c+1)*C_tile
                        for oi in range(O_tile):
                            kernel_tiles[ki, kj, c, :, oi] = F[o_start+oi, c_start:c_end, ki, kj]
            # Load bias
            bias_tile[:, 0] = bias[o_start:o_end]
            # For each row pair
            for r in range(n_rowpairs):
                toprow = 2*r
                # (1) Load input data
                for h in range(K+1):
                    for w in range(W):
                        for c in range(n_Ctiles):
                            input_tiles[h, c, :, w] = X[b, c_start:c_end, toprow+h, w]
                # (2) Do top row convolution
                out_tiles[0] = 0
                for ki in range(K):
                    for kj in range(K):
                        res_psum = np.zeros((O_tile, W_out))
                        for c in range(n_Ctiles):
                            res_psum += kernel_tiles[ki, kj, c].T@input_tiles[ki, c, :, kj:kj+W_out]
                        out_tiles[0] = res_psum
                # (3) Do bottom row convolution
                out_tiles[1] = 0
                for ki in range(K):
                    for kj in range(K):
                        res_psum = np.zeros((O_tile, W_out))
                        for c in range(n_Ctiles):
                            res_psum += kernel_tiles[ki, kj, c].T@input_tiles[ki+1, c, :, kj:kj+W_out]
                        out_tiles[1] = res_psum
                if (b == 0) and (r == 0) and (o == 0):
                    print(input_tiles[:3, 0, 0, :5])
                    print(X[0, 0, :3, :5])
                    print(kernel_tiles[:, :, 0, 0, 0])
                    print(F[0, 0, :, :])
                    print(X[0, 0, :3, :3]*F[0, 0, :, :])
                    print(np.sum(X[0, 0, :3, :3]*F[0, 0, :, :]))
                    print(np.sum(X[0, :, :3, :3]*F[0, :, :, :]))
                    print(out_tiles[0, 0, 0])
                # (4) Do pooling
                if pool_size == 1:
                    out_tiles[0] += bias_tile
                    out_tiles[1] += bias_tile
                    X_out[b, o_start:o_end, toprow, :] = out_tiles[0]
                    X_out[b, o_start:o_end, toprow+1, :] = out_tiles[1]
                elif pool_size == 2:
                    assert(False)

    return X_out

"""
A NumPy implementation of the forward pass for a convolutional layer.
"""
def conv_numpy(X, W, bias):
    out = None
    
    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, _, filter_height, filter_width = W.shape

    H_out = 1 + (input_height - filter_height)
    W_out = 1 + (input_width - filter_width)

    out = np.zeros((batch_size, out_channels, H_out, W_out))
    for b in range(batch_size):
        for c in range(out_channels):
            for i in range(H_out):
                for j in range(W_out):
                    x_ij = X[b, :, i : i + filter_height, j : j + filter_width]
                    out[b, c, i, j] = np.sum(x_ij * W[c]) + bias[c]

    return out

"""
A NumPy implementation of the forward pass for a max-pooling layer.
"""
def maxpool_numpy(X, pool_size):
    out = None

    batch_size, in_channels, input_height, input_width = X.shape
    
    H_out = 1 + (input_height - pool_size) // pool_size
    W_out = 1 + (input_width - pool_size) // pool_size

    out = np.zeros((batch_size, in_channels, H_out, W_out))

    for b in range(batch_size):
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * pool_size
                w_start = j * pool_size
                x_ij = X[b, :, h_start : h_start + pool_size, w_start : w_start + pool_size]
                out[b, :, i, j] = np.amax(x_ij, axis=(-1, -2))

    return out
