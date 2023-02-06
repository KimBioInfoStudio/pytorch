# Owner(s): ["module: onnx"]
from __future__ import annotations

import io
import os
import tempfile
import unittest

from typing import Any, Callable, Sequence, Tuple, Union

# import onnxruntime  # type: ignore[import]
import onnx.reference
import onnx_test_common

import torch
import transformers  # type: ignore[import]
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.nn import functional as F
from torch.onnx._internal import fx as fx_onnx
from torch.testing._internal import common_utils
from torch.utils import _pytree as pytree


class TestFxToOnnxWithOnnxRuntime(onnx_test_common._TestONNXRuntime):
    def setUp(self):
        super().setUp()
        self.opset_version = 17

    def _run_ort(
        self, onnx_model: Union[str, io.BytesIO], pytorch_inputs: Tuple[Any, ...]
    ) -> Sequence[Any]:
        session = onnx.reference.ReferenceEvaluator(onnx_model, verbose=5)
        input_names = session.input_names
        return session.run(
            None, {k: v.cpu().numpy() for k, v in zip(input_names, pytorch_inputs)}
        )

    def test_simple_function(self):
        def func(x):
            y = x + 1
            z = y.relu()
            return (y, z)

        tensor_x = torch.randn(1, 1, 2, dtype=torch.float32)

        self.run_test_with_fx_to_onnx_exporter(func, (tensor_x,))

    @unittest.skip("TypeError: export() got an unexpected keyword argument 'b'")
    def test_func_with_args_and_kwargs(self):
        def func(x, b=1.0):
            y = x + b
            z = y.relu()
            return (y, z)

        tensor_x = torch.randn(1, 1, 2, dtype=torch.float32)

        self.run_test_with_fx_to_onnx_exporter(func, (tensor_x,), {"b": 500.0})

    def test_mnist(self):
        class MNISTModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, 1, bias=True)
                self.conv2 = nn.Conv2d(32, 64, 3, 1, bias=True)
                self.fc1 = nn.Linear(9216, 128, bias=True)
                self.fc2 = nn.Linear(128, 10, bias=True)

            def forward(self, tensor_x: torch.Tensor):
                tensor_x = self.conv1(tensor_x)
                tensor_x = torch.sigmoid(tensor_x)
                tensor_x = self.conv2(tensor_x)
                tensor_x = torch.sigmoid(tensor_x)
                tensor_x = F.max_pool2d(tensor_x, 2)
                tensor_x = torch.flatten(tensor_x, 1)
                tensor_x = self.fc1(tensor_x)
                tensor_x = torch.sigmoid(tensor_x)
                output = self.fc2(tensor_x)
                return output

        tensor_x = torch.rand((64, 1, 28, 28), dtype=torch.float32)
        self.run_test_with_fx_to_onnx_exporter(MNISTModel(), (tensor_x,))

    # test single op with no kwargs
    def test_sigmoid(self):
        x = torch.randn(1, 4, 2, 3)

        class SigmoidModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sigmoid = torch.nn.Sigmoid()

            def forward(self, x):
                return self.sigmoid(x)

        self.run_test_with_fx_to_onnx_exporter(SigmoidModel(), (x,))

    # test single op with no kwargs
    def test_sigmoid_add(self):
        self.opset_version = 17
        # TODO(titaiwang): change to randn once it's ready
        x = torch.tensor([1.0, 2.0], dtype=torch.float)

        class SigmoidAddModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sigmoid = torch.nn.Sigmoid()

            def forward(self, x):
                x = torch.ops.aten.add(x, 1.0, alpha=2.0)
                return self.sigmoid(x)

        self.run_test_with_fx_to_onnx_exporter(SigmoidAddModel(), (x,))

    def test_gpt2_tiny(self):
        model_name = "sshleifer/tiny-gpt2"
        # Download pytorch model
        model = transformers.AutoModel.from_pretrained(model_name)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

        # Transform input tokens
        inputs = tokenizer("Hello world!", return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        onnx_model = fx_onnx.export_without_kwargs(
            model, self.opset_version, **inputs, use_binary_format=True
        )

        ref_outputs, _ = pytree.tree_flatten(model(**inputs, return_dict=False))
        ort_outputs = self._run_ort(onnx_model, (input_ids, attention_mask))
        assert len(ref_outputs) == len(ort_outputs)
        assert len(ref_outputs) == 5
        for ref_output, ort_output in zip(ref_outputs, ort_outputs):
            torch.testing.assert_allclose(ref_output, torch.tensor(ort_output))

    def _test_large_scale_exporter(
        self,
        model_name,
        create_model: Callable,
        create_args: Callable,
        create_pytorch_only_kwargs: Callable,
    ):
        """Test helper for large-scale exporter.

        Arguments:
            model_name: Name of the model. It used to name temporary files.
            create_model: A function that creates a model. It should always create the same model.
            create_args: A function that creates random input arguments for the model.
            create_pytorch_only_kwargs: A function that creates kwargs for calling PyTorch model with real tensors.

        This test contains several steps.
         1. Create a toy model.
         2. Save the toy's state (parameters) to a file. This is for simulating a checkpoint file.
         3. Load it back and export it to ONNX with large-scale exporter.
            All operations (including model loading) are done under
            FakeTensorMode so no real tensor is created and no real
            computation happens.
         4. The ONNX model generated in step 3 doesn't contain parameters,
            and this step adds them as external data and save a new ONNX model.
         5. Run PyTorch and ONNX models and compare their results.
        """

        # Create the toy model.
        model = create_model()

        with tempfile.NamedTemporaryFile(
            prefix=model_name, suffix=".pt"
        ) as tmp_file, tempfile.TemporaryDirectory(
            suffix="large_scale_export"
        ) as tmp_folder:
            # Dump state_dict to a file to simulate how HuggingFace model is initialized.
            # The file will be loaded via .load_state_dict(...)
            torch.save(model.state_dict(), tmp_file.name)

            ftm = FakeTensorMode(
                allow_non_fake_inputs=True, allow_fallback_kernels=False
            )
            ctx = fx_onnx.FxToOnnxContext()

            # The following coed block does several things.
            #  1. Create a model whose parameters and buffers are all FakeTensor's.
            #  2. Convert nn.Module into ONNX model without initializers.
            #  3. Record the file paths to find real initializers.
            with ftm, ctx:
                # Toy model with parameters and buffers as FakeTensor's.
                fake_model = create_model()
                fake_model.load_state_dict(torch.load(tmp_file.name))
                # Toy inputs as FakeTensor's.
                fake_args = create_args()
                # Export ONNX model without initializers while ctx.paths records
                # all files that contains real initializers.
                (onnx_model, _, _, _) = fx_onnx.export_without_parameters_and_buffers(
                    fake_model,
                    *fake_args,
                    use_binary_format=False,
                )

            # Tasks done by the following block.
            #  1. Iterate through all tensors stored in ctx.paths (the file content is loaded torch.load)
            #  2. If a tensor's name matches a "onnx_model"'s input name, an initializer is created and saved to
            #     a seperated folder.
            #  3. A new ONNX model is saved into file with the initializers saved in the previous step.
            #  4. ORT executes the new ONNX model and compares the results with the original GPT model.

            # Model saved to tmp_folder/onnx_model_location
            # Initializers are saved to tmp_folder/onnx_initializer_location/*.onnx
            onnx_model_location = model_name + "_external_data.onnx"
            onnx_initializer_location = model_name + "_initializers"
            fx_onnx.save_model_with_external_data(
                tmp_folder,
                onnx_model_location,
                onnx_initializer_location,
                tuple(ctx.paths),
                onnx_model,
            )

            # Generate random inputs.
            args = create_args()
            kwargs = create_pytorch_only_kwargs()
            # Original outputs.
            ref_outputs, _ = pytree.tree_flatten(model(*args, **kwargs))
            # ORT outputs.
            ort_outputs = self._run_ort(
                os.path.join(tmp_folder, onnx_model_location),
                (arg for arg in args if arg is not None),
            )

            assert len(ref_outputs) == len(ort_outputs)

            for ref_output, ort_output in zip(ref_outputs, ort_outputs):
                torch.testing.assert_allclose(ref_output, torch.tensor(ort_output))

    def test_large_scale_exporter_with_toy_mlp(self):
        class MLPModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc0 = nn.Linear(8, 8, bias=True)
                self.fc1 = nn.Linear(8, 4, bias=True)
                self.fc2 = nn.Linear(4, 2, bias=True)
                self.fc3 = nn.Linear(2, 2, bias=True)

            def forward(self, tensor_x: torch.Tensor):
                tensor_x = self.fc0(tensor_x)
                tensor_x = torch.sigmoid(tensor_x)
                tensor_x = self.fc1(tensor_x)
                tensor_x = torch.sigmoid(tensor_x)
                tensor_x = self.fc2(tensor_x)
                tensor_x = torch.sigmoid(tensor_x)
                output = self.fc3(tensor_x)
                return output

        def create_model():
            return MLPModel()

        def create_args():
            return (torch.rand((97, 8), dtype=torch.float32),)

        def create_pytorch_only_extra_kwargs():
            return {}

        self._test_large_scale_exporter(
            "toy_mlp1", create_model, create_args, create_pytorch_only_extra_kwargs
        )

    @unittest.skip("To pass this test, if-else conditions in GPT2 should be removed.")
    def test_large_scale_exporter_with_tiny_gpt2(self):
        model_name = "sshleifer/tiny-gpt2"

        def create_model():
            return transformers.AutoModel.from_pretrained(model_name)

        def create_args():
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            kwargs = tokenizer("Hello world!", return_tensors="pt")
            input_ids = kwargs["input_ids"]
            attention_mask = kwargs["attention_mask"]
            return input_ids, None, attention_mask

        def create_pytorch_only_extra_kwargs():
            return {"return_dict": False}

        self._test_large_scale_exporter(
            "tiny_gpt2", create_model, create_args, create_pytorch_only_extra_kwargs
        )


if __name__ == "__main__":
    common_utils.run_tests()
