import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


"""
A fused convolution - maxpool kernel that you need to implement for Part 2.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.
    pool_size: the size of the pool filter and pool stride.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: pool_size == 1 || pool_size == 2
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height // pool_size
out_pool_width = out_width // pool_size

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""

@nki.jit
def fused_conv2d_maxpool(X, F, bias, pool_size=1):
    # Process input sizes
    B, C, H, W = X.shape
    O, C_, K, K_ = F.shape
    O_ = bias.shape[0]

    assert(C == C_)
    assert(K == K_)
    assert(O == O_)
    assert(C % 128 == 0)
    assert(O % 128 == 0)

    # Define output dimensions
    H_out = H - K + 1
    W_out = W - K + 1
    H_pool = H_out // pool_size
    W_pool = W_out // pool_size
    
    # Can assume one PSUM bank can at least fit one row of the pixels
    assert(nl.tile_size.gemm_moving_fmax >= W_out)
 
    # Define tiling dimensions
    C_tile = nl.tile_size.pmax # 128
    n_Ctiles = C // C_tile

    O_tile = nl.tile_size.pmax # 128
    n_Otiles = O // O_tile

    n_rowpairs = H_out // 2

    datatype = X.dtype
    datasize = datatype.itemsize

    # Print things as needed
    '''
    print("\n")
    print(B, C, H, W)
    print(O, C, K)
    print(H_out, W_out)
    print(pool_size, H_pool, W_pool)
    print(C_tile, n_Ctiles)
    print(O_tile, n_Otiles)
    print(n_rowpairs)
    print(datatype, datasize)
    '''

    # Initialize output array
    X_out = nl.ndarray(shape=(B, O, H_pool, W_pool), dtype=datatype, buffer=nl.hbm)

    # Normal allocations
    input_tiles = nl.ndarray(
        (K+1, n_Ctiles, nl.par_dim(C_tile), W),
        dtype=datatype, buffer=nl.sbuf
    )
    kernel_tiles = nl.ndarray(
        (K, K, n_Ctiles, nl.par_dim(C_tile), O_tile),
        dtype=datatype, buffer=nl.sbuf
    )
    bias_tile = nl.ndarray(
        (nl.par_dim(O_tile), 1),
        dtype=datatype, buffer=nl.sbuf
    )
    out_tiles = nl.zeros(
        (2, nl.par_dim(O_tile), W_out),
        dtype=datatype, buffer=nl.sbuf
    ) 

    # Pooling allocations
    if pool_size == 2:
        pool_tile = nl.zeros((nl.par_dim(O_tile), W_pool), dtype=datatype, buffer=nl.sbuf)
        temp_pool = nl.zeros((nl.par_dim(O_tile), 4), dtype=datatype, buffer=nl.sbuf)
        i_0 = nl.arange(1)[:, None, None, None, None]
        i_1 = nl.arange(2)[None, :, None, None, None]
        i_2 = nl.arange(O_tile)[None, None, :, None, None]
        i_3 = nl.arange(W_pool)[None, None, None, :, None]
        i_4 = nl.arange(2)[None, None, None, None, :]

    # For each image
    for b in nl.sequential_range(B):
        # For each group of kernels
        for o in nl.sequential_range(n_Otiles): 
            o_start = o*O_tile
            o_end = (o+1)*O_tile
            # Load kernel
            for ki in nl.affine_range(K):
                for kj in nl.affine_range(K):
                    for c in nl.affine_range(n_Ctiles):
                        c_start = c*C_tile
                        c_end = (c+1)*C_tile
                        for oi in nl.affine_range(O_tile):
                            kernel_tiles[ki, kj, c, :, oi] = nl.load(F[o_start+oi, c_start:c_end, ki, kj])
            # Load bias
            bias_tile[...] = nl.load(bias[o_start:o_end])
            # For each row pair
            for r in nl.sequential_range(n_rowpairs):
                toprow = 2*r
                # (1) Load input data
                for h in nl.affine_range(K+1):
                    for w in nl.affine_range(W):
                        for c in nl.affine_range(n_Ctiles):
                            c_start = c*C_tile
                            c_end = (c+1)*C_tile
                            input_tiles[h, c, :, w] = nl.load(X[b, c_start:c_end, toprow+h, w])
                # (2) Do top row convolution
                out_tiles[0] = nl.multiply(out_tiles[0], 0)
                for ki in nl.sequential_range(K):
                    for kj in nl.sequential_range(K):
                        res_psum = nl.zeros((O_tile, W_out), nl.float32, buffer=nl.psum)
                        for c in nl.sequential_range(n_Ctiles):
                            res_psum[...] = res_psum + nl.matmul(kernel_tiles[ki, kj, c], input_tiles[ki, c, :, kj:kj+W_out], transpose_x=True)
                        temp = nl.copy(res_psum, dtype=out_tiles.dtype)
                        out_tiles[0] = nl.add(out_tiles[0], temp)
                # (3) Do bottom row convolution
                out_tiles[1] = nl.multiply(out_tiles[1], 0)
                for ki in nl.sequential_range(K):
                    for kj in nl.sequential_range(K):
                        res_psum = nl.zeros((O_tile, W_out), nl.float32, buffer=nl.psum)
                        for c in nl.sequential_range(n_Ctiles):
                            res_psum[...] = res_psum + nl.matmul(kernel_tiles[ki, kj, c], input_tiles[ki+1, c, :, kj:kj+W_out], transpose_x=True)
                        temp = nl.copy(res_psum, dtype=out_tiles.dtype)
                        out_tiles[1] = nl.add(out_tiles[1], temp)
                # (4) Do pooling
                if pool_size == 1:
                    # Don't do pooling
                    # Add bias
                    out_tiles[0] = nl.add(out_tiles[0], bias_tile)
                    out_tiles[1] = nl.add(out_tiles[1], bias_tile)
                    # Store results
                    nl.store(X_out[b, o_start:o_end, toprow, :], out_tiles[0])
                    nl.store(X_out[b, o_start:o_end, toprow+1, :], out_tiles[1])
                elif pool_size == 2:
                    # Do pooling
                    for w in nl.affine_range(W_pool):
                        temp_pool[:, 0] = out_tiles[0, :, 2*w]
                        temp_pool[:, 1] = out_tiles[1, :, 2*w]
                        temp_pool[:, 2] = out_tiles[0, :, 2*w+1]
                        temp_pool[:, 3] = out_tiles[1, :, 2*w+1]
                        pool_tile[:, w] = nl.max(temp_pool, axis=[1])
                    # Add bias
                    pool_tile[...] = nl.add(pool_tile, bias_tile)
                    # Store results
                    nl.store(X_out[b, o_start:o_end, r, :], pool_tile)

    return X_out

