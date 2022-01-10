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
# from utils import _compute_numerical_jacobian, _compute_numerical_batch_jacobian

class TestJacobian(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        
        self.dtype = 'float32'
        self.np_dtype = np.float32
        self.A = np.array([[1., 2.]]).astype('float32')
        self.B = np.array([[1., 2.], [2., 1.]]).astype('float32')
        self.numerical_delta = 1e-4
        self.rtol = 1e-3
        self.atol = 1e-3

    def test_standard_elementwise_function(self):
        def func(x):
            return paddle.multiply(x, x)
        
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
        rows = exe.run(main, feed={'x':self.A}, fetch_list=[rows])
        print(rows)
        # JJ = _compute_numerical_jacobian(
        #     func, self.A, self.numerical_delta, self.np_dtype)[0][0]

if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()