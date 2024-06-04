# passthrough_kernel/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
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
    lineWidthInBytes =(int)( N //(1000000/4))   # chop input in 4 sub-tensors
    lineWidthInInt32s = lineWidthInBytes // 4

    @device(AIEDevice.npu1_4col)
    def device_body():
        # define types
        memRef_ty = T.memref(lineWidthInBytes, T.ui8())
        
        # AIE Core Function declarations
        passThroughLine = external_func(
            "passThroughLine", inputs=[memRef_ty, memRef_ty, T.i32()]
        )

        # Tile declarations
        ShimTile0 = tile(0, 0)
        ComputeTile0_2 = tile(0, 2)

        ShimTile1 = tile(1, 0)
        ComputeTile1_2 = tile(1, 2)

        ShimTile2 = tile(2, 0)
        ComputeTile2_2 = tile(2, 2)
        
        ShimTile3 = tile(3, 0)
        ComputeTile3_2 = tile(3, 2)


        # AIE-array data movement with object fifos
        of_in0 = object_fifo("in0", ShimTile0, ComputeTile0_2, 2, memRef_ty)
        of_out0 = object_fifo("out0", ComputeTile0_2, ShimTile0, 2, memRef_ty)

        of_in1 = object_fifo("in1", ShimTile1, ComputeTile1_2, 2, memRef_ty)
        of_out1 = object_fifo("out1", ComputeTile1_2, ShimTile1, 2, memRef_ty)
        
        of_in2 = object_fifo("in2", ShimTile2, ComputeTile2_2, 2, memRef_ty)
        of_out2 = object_fifo("out2", ComputeTile2_2, ShimTile2, 2, memRef_ty)
        
        of_in3 = object_fifo("in3", ShimTile3, ComputeTile3_2, 2, memRef_ty)
        of_out3 = object_fifo("out3", ComputeTile3_2, ShimTile3, 2, memRef_ty)
        # Set up compute tiles

        # Compute tile 2
        @core(ComputeTile0_2, "passThrough.cc.o")
        def core_body():
            for _ in for_(sys.maxsize):
                elemOut = of_out0.acquire(ObjectFifoPort.Produce, 1)
                elemIn = of_in0.acquire(ObjectFifoPort.Consume, 1)
                call(passThroughLine, [elemIn, elemOut, lineWidthInBytes])
                of_in0.release(ObjectFifoPort.Consume, 1)
                of_out0.release(ObjectFifoPort.Produce, 1)
                yield_([])

        # Compute tile 1 2
        @core(ComputeTile1_2, "passThrough.cc.o")
        def core_body():
            for _ in for_(sys.maxsize):
                elemOut = of_out1.acquire(ObjectFifoPort.Produce, 1)
                elemIn = of_in1.acquire(ObjectFifoPort.Consume, 1)
                call(passThroughLine, [elemIn, elemOut, lineWidthInBytes])
                of_in1.release(ObjectFifoPort.Consume, 1)
                of_out1.release(ObjectFifoPort.Produce, 1)
                yield_([])

        # Compute tile 2 2
        @core(ComputeTile2_2, "passThrough.cc.o")
        def core_body():
            for _ in for_(sys.maxsize):
                elemOut = of_out2.acquire(ObjectFifoPort.Produce, 1)
                elemIn = of_in2.acquire(ObjectFifoPort.Consume, 1)
                call(passThroughLine, [elemIn, elemOut, lineWidthInBytes])
                of_in2.release(ObjectFifoPort.Consume, 1)
                of_out2.release(ObjectFifoPort.Produce, 1)
                yield_([])

        # Compute tile 3 2
        @core(ComputeTile3_2, "passThrough.cc.o")
        def core_body():
            for _ in for_(sys.maxsize):
                elemOut = of_out3.acquire(ObjectFifoPort.Produce, 1)
                elemIn = of_in3.acquire(ObjectFifoPort.Consume, 1)
                call(passThroughLine, [elemIn, elemOut, lineWidthInBytes])
                of_in3.release(ObjectFifoPort.Consume, 1)
                of_out3.release(ObjectFifoPort.Produce, 1)
                yield_([])



        #    print(ctx.module.operation.verify())

        tensorSize = N
        tensorSizeInInt32s = tensorSize // 4
        tensor_ty = T.memref(tensorSizeInInt32s*4, T.i32()) #4 col

        @FuncOp.from_py_func(tensor_ty, tensor_ty, tensor_ty)
        def sequence(inTensor, outTensor, notUsed):

            npu_dma_memcpy_nd(
                metadata="in0",
                bd_id=0,
                mem=inTensor,
                sizes=[1, 1, 1, tensorSizeInInt32s],
            )
            npu_dma_memcpy_nd(
                metadata="out0",
                bd_id=1,
                mem=outTensor,
                sizes=[1, 1, 1, tensorSizeInInt32s],
            )

            npu_dma_memcpy_nd(
                metadata="in1",
                bd_id=2,
                mem=inTensor,
                offsets=[0, 0, 0, tensorSizeInInt32s],
                sizes=[1, 1, 1, tensorSizeInInt32s],
            )
            npu_dma_memcpy_nd(
                metadata="out1",
                bd_id=3,
                mem=outTensor,
                offsets=[0, 0, 0, tensorSizeInInt32s],
                sizes=[1, 1, 1, tensorSizeInInt32s],
            )

            npu_dma_memcpy_nd(
                metadata="in2",
                bd_id=4,
                mem=inTensor,
                offsets=[0, 0, 0, 2*tensorSizeInInt32s],
                sizes=[1, 1, 1, tensorSizeInInt32s],
            )
            npu_dma_memcpy_nd(
                metadata="out2",
                bd_id=5,
                mem=outTensor,
                offsets=[0, 0, 0, 2*tensorSizeInInt32s],
                sizes=[1, 1, 1, tensorSizeInInt32s],
            )

            npu_dma_memcpy_nd(
                metadata="in3",
                bd_id=6,
                mem=inTensor,
                offsets=[0, 0, 0, 3*tensorSizeInInt32s],
                sizes=[1, 1, 1, tensorSizeInInt32s],
            )
            npu_dma_memcpy_nd(
                metadata="out3",
                bd_id=7,
                mem=outTensor,
                offsets=[0, 0, 0, 3*tensorSizeInInt32s],
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
