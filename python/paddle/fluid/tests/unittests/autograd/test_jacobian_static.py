# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import numpy as np
import paddle
import paddle.fluid as fluid
from utils import _compute_numerical_jacobian, _compute_numerical_batch_jacobian

def approx_jacobian(f, xs, dtype, eps=1e-5, batch=False):
    r"""Computes an approximate Jacobian matrix of a multi-valued function 
    using finite differences.
    
    The function input is required to be an np array or a list of list of np 
    arrays. 
    """
    def flatten(xs):
        if isinstance(xs, np.ndarray):
            flattened = xs.flatten()
        else:
            flattened = np.concatenate([x.flatten() for x in xs])
        return flattened

    def x_like(x, orig_x):
        return x.reshape(orig_x.shape)

    def _f(x):
        if multi_inps:
            _xs = np.split(x, splits)
            _xs = [x_like(_x, _o) for _x, _o in zip(_xs, xs)]      
            outs = f(_xs)
        else:
            outs = f(x)
        return flatten(outs)

    multi_inps = False if isinstance(xs, np.ndarray) else True
    xdim = xs.size if isinstance(xs, np.ndarray) else sum(x.size for x in xs)
    splits = []

    x = flatten(xs)
    if multi_inps:
        split = 0
        for inp in xs:
            split += inp.size
            splits.append(split)

    ds = eps * np.eye(xdim, dtype=dtype)
    
    fprimes_by_x = [0.5 * (_f(x + d) - _f(x - d)) / eps for d in ds]
    return np.stack(fprimes_by_x).T

class TestJacobian(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        paddle.enable_static()
        self.np_dtype = np.float32
        self.A = np.array([[1., 2.]]).astype('float32')
        self.B = np.array([[1., 2.], [2., 1.]]).astype('float32')
        self.eps = 1e-5
        self.rtol = 1e-3
        self.atol = 1e-3

    def test_standard_elementwise_function(self):
        def func(x):
            return paddle.multiply(x, x)
        
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(main, startup):
            x = paddle.static.data(name='x', shape=[1, 2], dtype=self.np_dtype)
            JJ = paddle.autograd.functional.Jacobian(func, x, batch=False)
            nrow, ncol = JJ.shape()
            full_jacobian = JJ[:]
        place = fluid.CUDAPlace(0)
        exe = fluid.Executor(place)
        exe.run(startup)
        pd_jacobians = exe.run(main, feed={'x':self.A}, fetch_list=[full_jacobian])
        print(pd_jacobians)

        def np_func(x):
            return x * x
        np_jacobians = approx_jacobian(np_func, self.A, self.np_dtype, self.eps)
        print(np_jacobians)

def test_standard_tensor_function(self):
        def func(x):
            return paddle.matmul(x, x)
        
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(main, startup):
            x = paddle.static.data(name='x', shape=[1, 2], dtype=self.dtype)
            JJ = paddle.autograd.functional.Jacobian(func, x, batch=True)
            nrow, ncol = JJ.shape()
            rows = [JJ[i] for i in range(nrow)]
        place = fluid.CUDAPlace(0)
        exe = fluid.Executor(place)
        exe.run(startup)
        rows = exe.run(main, feed={'x':self.B}, fetch_list=[rows])

if __name__ == "__main__":
    unittest.main()