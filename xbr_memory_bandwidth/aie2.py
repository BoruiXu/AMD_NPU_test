# passthrough_kernel/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license infromation.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates

import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *
from aie.extras.context import mlir_mod_ctx

import aie.utils.trace as trace_utils


def passthroughKernel(vector_size, trace_size):
    N =(int) (vector_size/4)
    lineWidthInBytes =(int)( N //(1000000/4))   # each chunk 1000 elements
    lineWidthInInt32s = lineWidthInBytes // 4

    @device(AIEDevice.npu1_4col)
    def device_body():
        # define types
        memRef_ty = T.memref(lineWidthInBytes, T.ui8())
        inerface_memRef_ty = T.memref(2*lineWidthInBytes, T.ui8())
        
        # AIE Core Function declarations
        passThroughLine = external_func(
            "passThroughLine", inputs=[memRef_ty, memRef_ty, T.i32()]
        )

        # Tile declarations
        ShimTile0 = tile(0, 0)
        MemTile0_1 = tile(0,1)
        ComputeTile0_2 = tile(0, 2)
        ComputeTile0_3 = tile(0, 3)

        ShimTile1 = tile(1, 0)
        MemTile1_1 = tile(1,1)
        ComputeTile1_2 = tile(1, 2)
        ComputeTile1_3 = tile(1, 3)

        ShimTile2 = tile(2, 0)
        MemTile2_1 = tile(2,1)
        ComputeTile2_2 = tile(2, 2)
        ComputeTile2_3 = tile(2, 3)
        
        ShimTile3 = tile(3, 0)
        MemTile3_1 = tile(3,1)
        ComputeTile3_2 = tile(3, 2)
        ComputeTile3_3 = tile(3, 3)


        
        buffer_in_dic = {}
        buffer_out_dic = {}
        buffer_in_name = []
        buffer_out_name = []

        buffer_compute_in_dic = {}
        buffer_compute_out_dic = {}
        buffer_compute_in_name = []
        buffer_compute_out_name = []
        
        buffer_compute2_in_dic = {}
        buffer_compute2_out_dic = {}
        buffer_compute2_in_name = []
        buffer_compute2_out_name = []
        
        interface_list = [ShimTile0,ShimTile1,ShimTile2,ShimTile3]
        memory_list = [MemTile0_1,MemTile1_1,MemTile2_1,MemTile3_1]
        compute_list = [ComputeTile0_2,ComputeTile1_2,ComputeTile2_2,ComputeTile3_2]
        compute2_list = [ComputeTile0_3,ComputeTile1_3,ComputeTile2_3,ComputeTile3_3]
        


        for i in range(4):
            #interface to memory tile
            in_name = "in"+str(i)
            buffer_in_name.append(in_name)

            out_name = "out"+str(i)
            buffer_out_name.append(out_name)

            buffer_in_dic[in_name] = object_fifo(in_name, interface_list[i], memory_list[i], 2, inerface_memRef_ty)
            buffer_out_dic[out_name] = object_fifo(out_name, memory_list[i], interface_list[i], 2, inerface_memRef_ty)

            #memory tile and compute1 tile
            to_compute1 = "to_compute1_"+str(i)
            buffer_compute_in_name.append(to_compute1)

            from_compute1 = "from_compute1_"+str(i)
            buffer_compute_out_name.append(from_compute1)

            buffer_compute_in_dic[to_compute1] = object_fifo(to_compute1, memory_list[i], compute_list[i], 2, memRef_ty)
            buffer_compute_out_dic[from_compute1] = object_fifo(from_compute1, compute_list[i], memory_list[i], 2, memRef_ty)

            #memory tile and compute2 tile
            to_compute2 = "to_compute2_"+str(i)
            buffer_compute2_in_name.append(to_compute2)

            from_compute2 = "from_compute2_"+str(i)
            buffer_compute2_out_name.append(from_compute2)

            buffer_compute2_in_dic[to_compute2] = object_fifo(to_compute2, memory_list[i], compute2_list[i], 2, memRef_ty)
            buffer_compute2_out_dic[from_compute2] = object_fifo(from_compute2, compute2_list[i], memory_list[i], 2, memRef_ty)

            #link
            object_fifo_link(buffer_in_dic[in_name], [ buffer_compute_in_dic[to_compute1], buffer_compute2_in_dic[to_compute2]])
            object_fifo_link([buffer_compute_out_dic[from_compute1],buffer_compute2_out_dic[from_compute2]],buffer_out_dic[out_name])



        # Set up compute1 tiles
        for i in range(4):
            @core(compute_list[i], "passThrough.cc.o")
            def core_body():
                for _ in for_(sys.maxsize):
                    elemOut = buffer_compute_out_dic[buffer_compute_out_name[i]].acquire(ObjectFifoPort.Produce, 1)
                    elemIn = buffer_compute_in_dic[buffer_compute_in_name[i]].acquire(ObjectFifoPort.Consume, 1)
                    call(passThroughLine, [elemIn, elemOut, lineWidthInBytes])
                    buffer_compute_in_dic[buffer_compute_in_name[i]].release(ObjectFifoPort.Consume, 1)
                    buffer_compute_out_dic[buffer_compute_out_name[i]].release(ObjectFifoPort.Produce, 1)
                    yield_([])

        # Set up compute2 tiles
        for i in range(4):
            @core(compute2_list[i], "passThrough.cc.o")
            def core_body():
                for _ in for_(sys.maxsize):
                    elemOut = buffer_compute2_out_dic[buffer_compute2_out_name[i]].acquire(ObjectFifoPort.Produce, 1)
                    elemIn = buffer_compute2_in_dic[buffer_compute2_in_name[i]].acquire(ObjectFifoPort.Consume, 1)
                    call(passThroughLine, [elemIn, elemOut, lineWidthInBytes])
                    buffer_compute2_in_dic[buffer_compute2_in_name[i]].release(ObjectFifoPort.Consume, 1)
                    buffer_compute2_out_dic[buffer_compute2_out_name[i]].release(ObjectFifoPort.Produce, 1)
                    yield_([])


        #    print(ctx.module.operation.verify())

        tensorSize = N
        tensorSizeInInt32s = tensorSize // 4
        tensor_ty = T.memref(tensorSizeInInt32s*4, T.i32()) #4 col

        @FuncOp.from_py_func(tensor_ty, tensor_ty, tensor_ty)
        def sequence(inTensor, outTensor, notUsed):

            for i in range(4):
                npu_dma_memcpy_nd(
                    metadata=buffer_in_name[i],
                    bd_id=2*i,
                    mem=inTensor,
                    offsets=[0, 0, 0, i*tensorSizeInInt32s],
                    sizes=[1, 1, 1, tensorSizeInInt32s],
                )
                npu_dma_memcpy_nd(
                    metadata=buffer_out_name[i],
                    bd_id=2*i+1,
                    mem=outTensor,
                    offsets=[0, 0, 0, i*tensorSizeInInt32s],
                    sizes=[1, 1, 1, tensorSizeInInt32s],
                )

            npu_sync(column=0, row=0, direction=0, channel=0)
            npu_sync(column=1, row=0, direction=0, channel=0)
            npu_sync(column=2, row=0, direction=0, channel=0)
            npu_sync(column=3, row=0, direction=0, channel=0)


try:
    vector_size = int(sys.argv[1])
    if vector_size % 64 != 0 or vector_size < 512:
        print("Vector size must be a multiple of 64 and greater than or equal to 512")
        raise ValueError
    trace_size = 0 if (len(sys.argv) != 3) else int(sys.argv[2])
except ValueError:
    print("Argument has inappropriate value")
with mlir_mod_ctx() as ctx:
    passthroughKernel(vector_size, trace_size)
    print(ctx.module)
