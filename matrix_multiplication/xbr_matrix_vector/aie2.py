#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.
import argparse
from aie.extras.context import mlir_mod_ctx

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *


def my_matmul(M = 288, K = 288):
    #M = 288
    #K = 288
    m = 64
    k = 64
    word_size_in = 2
    word_size_out = 4

    n_cores = 8
    n_cols = 4
    cores_div_col = n_cores//n_cols

    A_sz_in_i32s = M * K * word_size_in // 4
    B_sz_in_i32s = K * word_size_in // 4
    C_sz_in_bytes = M * word_size_out
    C_sz_in_i32s = C_sz_in_bytes // 4
    C_sz_div_n_cores_in_i32s = C_sz_in_i32s // n_cores

    M_div_m = M // m
    M_div_m_div_n_cores = M // (m * n_cores)
    K_div_k = K // k

    K_in_i32s = K * word_size_in // 4
    k_in_i32s = k * word_size_in // 4
    m_in_i32s = m * word_size_in // 4
    m_x_k_in_i32s = m * k * word_size_in // 4
    m_x_K_in_i32s = m * K * word_size_in // 4

    vectorized = True

    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1_4col)
        def device_body():
            memRef_inA_ty = T.memref(m * k * cores_div_col, T.bf16()) #4 compute tile in one col
            memRef_inB_ty = T.memref(k, T.bf16())
            memRef_C_ty = T.memref(m, T.f32())
            memRef_outC_ty = T.memref(m*cores_div_col, T.f32())
            memRef_A_ty = T.memref(m, k, T.bf16())

            # AIE Core Function declarations
            zero_scalar = external_func("zero_scalar_f32", inputs=[memRef_C_ty])
            zero = external_func("zero_vectorized_f32", inputs=[memRef_C_ty])
            matvec_scalar = external_func(
                "matvec_scalar_bf16_f32",
                inputs=[memRef_A_ty, memRef_inB_ty, memRef_C_ty],
            )
            matvec = external_func(
                "matvec_vectorized_bf16_f32",
                inputs=[memRef_A_ty, memRef_inB_ty, memRef_C_ty],
            )

            # Tile declarations
            ShimTile0 = tile(0, 0)
            ShimTile1 = tile(1, 0)
            ShimTile2 = tile(2, 0)
            ShimTile3 = tile(3, 0)
            ShimTiles = [ShimTile0, ShimTile1, ShimTile2, ShimTile3]
            MemTile0 = tile(0, 1)
            MemTile1 = tile(1, 1)
            MemTile2 = tile(2, 1)
            MemTile3 = tile(3, 1)
            MemTiles = [MemTile0, MemTile1, MemTile2, MemTile3]
            
            ComputeTile0_2 = tile(0, 2)
            ComputeTile0_3 = tile(0, 3)
            ComputeTile0_4 = tile(0, 4)
            ComputeTile0_5 = tile(0, 5)
            
            ComputeTile1_2 = tile(1, 2)
            ComputeTile1_3 = tile(1, 3)
            ComputeTile1_4 = tile(1, 4)
            ComputeTile1_5 = tile(1, 5)
            
            ComputeTile2_2 = tile(2, 2)
            ComputeTile2_3 = tile(2, 3)
            ComputeTile2_4 = tile(2, 4)
            ComputeTile2_5 = tile(2, 5)
            
            ComputeTile3_2 = tile(3, 2)
            ComputeTile3_3 = tile(3, 3)
            ComputeTile3_4 = tile(3, 4)
            ComputeTile3_5 = tile(3, 5)
            

            #memA0 for computetile0_2 ~ computetile0_5
            cores = [ComputeTile0_2, ComputeTile0_3, ComputeTile0_4, ComputeTile0_5,
                     ComputeTile1_2, ComputeTile1_3, ComputeTile1_4, ComputeTile1_5,
                     ComputeTile2_2, ComputeTile2_3, ComputeTile2_4, ComputeTile2_5,
                     ComputeTile3_2, ComputeTile3_3, ComputeTile3_4, ComputeTile3_5]
            

            memA_fifo_names = ["memA0", "memA1", "memA2", "memA3"]
            memA_fifos = {}
            inA_fifo_names = ["inA02", "inA03", "inA04", "inA05",
                              "inA12", "inA13", "inA14", "inA15", 
                              "inA22", "inA23", "inA24", "inA25",
                              "inA32", "inA33", "inA34", "inA35"]
            inA_fifos = {}
            inB_fifo_names = ["inB"]
            inB_fifos = {}
            memC_fifo_names = ["memC0", "memC1", "memC2", "memC3"]
            memC_fifos = {}
            outC_fifo_names = ["outC02", "outC03", "outC04", "outC05",
                               "outC12", "outC13", "outC14", "outC15", 
                               "outC22", "outC23", "outC24", "outC25",
                               "outC32", "outC33", "outC34", "outC35"]
            outC_fifos = {}

            # AIE-array data movement with object fifos
            # Input A
            for i in range(n_cols):
                memA_fifos[memA_fifo_names[i]] = object_fifo(
                    memA_fifo_names[i],
                    ShimTiles[i],
                    MemTiles[i],
                    2,
                    memRef_inA_ty,
                )

                # 4 compute tiles
                tmp_list = []
                for j in range(cores_div_col):
                    inA_fifos[inA_fifo_names[i*cores_div_col+j]] = object_fifo( inA_fifo_names[i*cores_div_col+j], 
                                                                    MemTiles[i], cores[i*cores_div_col+j], 
                                                                    2, memRef_A_ty,
                                                                    [
                                                                        (k//2 , 2),
                                                                        (m, k),
                                                                        (2, 1),
                                                                    ],  # transpose at 4-byte (2xbf16) granularity
                                                                )
                    tmp_list.append(inA_fifos[inA_fifo_names[i*cores_div_col+j]])
                
                #object_fifo_link(memA_fifos[memA_fifo_names[i]], inA_fifos[inA_fifo_names[i]])
                #distribute
                object_fifo_link(memA_fifos[memA_fifo_names[i]], tmp_list)

            # Input B
            inB_fifos[inB_fifo_names[0]] = object_fifo(
                inB_fifo_names[0],
                ShimTiles[1 % n_cores],
                cores[0:n_cores],
                2,
                memRef_inB_ty,
            )

            # Output C
            for i in range(n_cols):
                tmp_list = []
                for j in range(cores_div_col):
                    outC_fifos[outC_fifo_names[i*cores_div_col+j]] = object_fifo(outC_fifo_names[i*cores_div_col+j],
                                                                     cores[i*cores_div_col+j],MemTiles[i],
                                                                     2,memRef_C_ty,
                                                                     )
                    tmp_list.append(outC_fifos[outC_fifo_names[i*cores_div_col+j]])

                memC_fifos[memC_fifo_names[i]] = object_fifo(
                    memC_fifo_names[i],
                    MemTiles[i],
                    ShimTiles[i],
                    2,
                    memRef_outC_ty,
                )
                #join
                object_fifo_link(tmp_list,memC_fifos[memC_fifo_names[i]])


            # Set up compute tiles
            for i in range(n_cores):
                # Compute tile i
                @core(cores[i], "mv.o")
                def core_body():
                    for _ in for_(0xFFFFFFFF):
                        elem_out = outC_fifos[outC_fifo_names[i]].acquire(
                            ObjectFifoPort.Produce,
                            1,
                        )
                        call(zero, [elem_out])

                        for _ in for_(K_div_k):
                            elem_in_a = inA_fifos[inA_fifo_names[i]].acquire(
                                ObjectFifoPort.Consume,
                                1,
                            )
                            elem_in_b = inB_fifos[inB_fifo_names[0]].acquire(
                                ObjectFifoPort.Consume,
                                1,
                            )
                            call(matvec, [elem_in_a, elem_in_b, elem_out])
                            inA_fifos[inA_fifo_names[i]].release(
                                ObjectFifoPort.Consume,
                                1,
                            )
                            inB_fifos[inB_fifo_names[0]].release(
                                ObjectFifoPort.Consume,
                                1,
                            )
                            yield_([])

                        outC_fifos[outC_fifo_names[i]].release(
                            ObjectFifoPort.Produce,
                            1,
                        )
                        yield_([])

            # To/from AIE-array data movement

            @FuncOp.from_py_func(
                T.memref(A_sz_in_i32s, T.i32()),
                T.memref(B_sz_in_i32s, T.i32()),
                T.memref(C_sz_in_i32s, T.i32()),
            )
            def sequence(A, B, C):
                npu_dma_memcpy_nd(
                    metadata=inB_fifo_names[0],
                    bd_id=2,
                    mem=B,
                    sizes=[M_div_m_div_n_cores, 1, 1, K_in_i32s],
                    strides=[0, 0, 0],
                )
                for i in range(n_cols):
                    A_offset = cores_div_col * i * M_div_m_div_n_cores * m * K * word_size_in // 4
                    C_offset = cores_div_col * i * M_div_m_div_n_cores * m * word_size_out // 4
                    npu_dma_memcpy_nd(
                        metadata=memA_fifo_names[i],
                        bd_id=1,
                        mem=A,
                        offsets=[0, 0, 0, A_offset],
                        sizes=[M_div_m_div_n_cores, K_div_k, cores_div_col*m, k_in_i32s],
                        strides=[cores_div_col*m_x_K_in_i32s, k_in_i32s, K_in_i32s],
                    )
                    npu_dma_memcpy_nd(
                        metadata=memC_fifo_names[i],
                        bd_id=0,
                        mem=C,
                        offsets=[0, 0, 0, C_offset],
                        sizes=[1, 1, 1, cores_div_col*C_sz_div_n_cores_in_i32s],
                        strides=[0, 0, 0],
                    )

                for i in range(n_cores):
                    npu_sync(column=i, row=0, direction=0, channel=0)

    print(ctx.module)


argparser = argparse.ArgumentParser(
    prog="AIE Matrix Multiplication MLIR Design (Whole Array)",
    description="Emits MLIR code for a matrix multiplication design of the given input size",
)
argparser.add_argument("-M", type=int, default=288)
argparser.add_argument("-K", type=int, default=288)
argparser.add_argument("-N", type=int, default=288)
args = argparser.parse_args()
my_matmul(args.M, args.K)
