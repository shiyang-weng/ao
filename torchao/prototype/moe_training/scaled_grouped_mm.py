# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional, Tuple

import torch

from torchao.float8.config import ScalingGranularity
from torchao.float8.float8_utils import tensor_to_scale, to_fp8_saturated
from torchao.prototype.moe_training.kernels import (
    triton_fp8_col_major_jagged_colwise_scales,
    triton_fp8_row_major_jagged_rowwise_scales,
)
from torchao.prototype.moe_training.utils import (
    _is_column_major,
)
from torchao.prototype.mx_formats.mx_tensor import to_mx

logger: logging.Logger = logging.getLogger(__name__)


def _scaled_grouped_mm(
    A: torch.Tensor,
    B_t: torch.Tensor,
    offs: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = torch.bfloat16,
) -> torch.Tensor:
    """
    This function performs dynamic float8 quantization with row-wise scaling
    on the input tensors A and B, then performs a scaled grouped GEMM and returns the results.

    Args:
        A (bf16/float32 torch.Tensor): The first high-precision input tensor, which must be a 2D tensor of shape (M * num_groups, K)
            and in row-major memory layout.
        B_t (bf16/float32 torch.Tensor): The second high-precision input tensor which must be 3D, which must be shape (E, K, N)
            and in column-major memory layout.
        offs (int32 torch.Tensor): The offsets to use to mark the starting index of each group along dim0 of the A tensor.
        out_dtype (Optional[torch.dtype]): The dtype of the output tensor. Currently only torch.bfloat16 is supported.
    """
    # TODO: Remove once prototype is more mature. This is currently very useful for development and debugging.
    logger.info("Using scaled_grouped_mm")
    return _Float8GroupedMM.apply(
        A,
        B_t,
        offs,
        out_dtype,
    )


class _Float8GroupedMM(torch.autograd.Function):
    """Differentiable implementation of grouped GEMM with dynamic float8 quantization."""

    @staticmethod
    def forward(
        ctx,
        A: torch.Tensor,
        B_t: torch.Tensor,
        offs: Optional[torch.Tensor] = None,
        out_dtype: Optional[torch.dtype] = torch.bfloat16,
    ) -> torch.Tensor:
        # torchao _scaled_grouped_mm only supports A=2D|3D and B=3D.
        assert A.ndim == 2 or A.ndim == 3, "A must be 2D or 3D"
        assert B_t.ndim == 3, "B must be 3D"

        assert A.size(-1) % 16 == 0, (
            f"A must have a last dim divisible by 16, but got shape: {A.shape}"
        )
        assert B_t.size(-2) % 16 == 0 and B_t.size(-1) % 16 == 0, (
            f"B must have last 2 dims divisible by 16, but got shape: {B_t.shape}"
        )

        # Assert input tensors are in high-precision dtypes.
        assert A.dtype == torch.float32 or A.dtype == torch.bfloat16, (
            "A must be float32 or bfloat16"
        )
        assert B_t.dtype == torch.float32 or B_t.dtype == torch.bfloat16, (
            "B must be float32 or bfloat16"
        )
        assert offs is None or offs.dtype == torch.int32, (
            "offs must be int32 tensor or None"
        )

        # Assert A and B dims are compatible for a scaled grouped GEMM.
        assert A.size(-1) == B_t.size(-2), (
            f"shape {A.shape} and {B_t.shape} are not compatible for _scaled_grouped_mm"
        )

        # The left operand in the scaled grouped GEMM must be row-major due to hardware requirements.
        assert not _is_column_major(A), "A must be row-major"

        # Due to hardware requirements, the right operand in a scaled grouped GEMM must be column-major.
        if not _is_column_major(B_t):
            # FSDP will complain if B_t (weights) is not contiguous, we can't require B_t to be column-major.
            # TODO: figure out better solution than transposing for each forward pass.
            B_t = B_t.transpose(-2, -1).contiguous().transpose(-2, -1)

        # Convert high precision input tensor to float8, row-major for left operand of grouped GEMM.
        # A shape: (M, K) or (B, M, K)
        # A_scales shape: (M,1) or (B, M, 1)
        A_scales = tensor_to_scale(
            A,
            torch.float8_e4m3fn,
            scaling_granularity=ScalingGranularity.AXISWISE,
            axiswise_dim=-1,
            round_scales_to_power_of_2=True,
        )
        A_scaled = A.to(torch.float32) * A_scales
        A_fp8_row_major = to_fp8_saturated(A_scaled, torch.float8_e4m3fn)

        # Convert B to float8, column-major for right operand of grouped GEMM.
        # B shape: (E, K, N)
        # B scales must be computed rowwise keeping the outer/final dim, so:
        # B_scales shape: (E, 1, N)
        B_t_scales = tensor_to_scale(
            B_t,
            torch.float8_e4m3fn,
            scaling_granularity=ScalingGranularity.AXISWISE,
            axiswise_dim=-2,
            round_scales_to_power_of_2=True,
        )
        B_t_scaled = B_t.to(torch.float32) * B_t_scales
        B_t_fp8_col_major = to_fp8_saturated(B_t_scaled, torch.float8_e4m3fn)

        # Precompute non-transposed B column-major for backward, to save memory by storing the
        # low precision B tensor instead of the high precision B tensor.
        # In the backward this is needed for grad_A: grad_output @ B.
        B = B_t.contiguous().transpose(-2, -1)

        # - B shape: (E, K, N)
        # - B scales must be computed rowwise keeping the outer/final dim, so:
        # - B_scale shape: (E, 1, N)
        B_scales = tensor_to_scale(
            B,
            torch.float8_e4m3fn,
            scaling_granularity=ScalingGranularity.AXISWISE,
            axiswise_dim=-2,
            round_scales_to_power_of_2=True,
        )
        B_scaled = B.to(torch.float32) * B_scales
        B_fp8_col_major = to_fp8_saturated(B_scaled, torch.float8_e4m3fn)

        # Store what we need for backward.
        ctx.save_for_backward(A, B_fp8_col_major, B_scales, offs)
        ctx.out_dtype = out_dtype

        # Perform scaled grouped GEMM and return result.
        # output shape: scaled grouped mm of (M,K) @ (B,K,N) = (M,N)
        assert not _is_column_major(A_fp8_row_major), (
            "A must be row-major for output = A @ B"
        )
        assert _is_column_major(B_t_fp8_col_major), (
            "B must be column-major for output = A @ B"
        )

        # Squeeze empty dims out of scales, to comply with grouped mm API.
        # A_scales shape: (M,1) or (B, M, 1)
        # B_t_scales shape: (E, 1, N)
        A_scales = A_scales.squeeze(-1)
        B_t_scales = B_t_scales.squeeze(1)
        return torch._scaled_grouped_mm(
            A_fp8_row_major,
            B_t_fp8_col_major,
            A_scales.reciprocal(),  # Reciprocals are needed for rescaling the output.
            B_t_scales.reciprocal(),
            offs,
            out_dtype=out_dtype,
            use_fast_accum=True,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        A, B_fp8_col_major, B_scales, offs = ctx.saved_tensors
        out_dtype = ctx.out_dtype

        # Convert grad_output to float8, row-major for left operand of grouped GEMM
        # needed for grad_A: grad_output @ B
        #
        # grad_output shape: (M, N)
        # grad_output_scale shape: (M, 1)
        grad_output_scales = tensor_to_scale(
            grad_output,
            torch.float8_e4m3fn,
            scaling_granularity=ScalingGranularity.AXISWISE,
            axiswise_dim=-1,
            round_scales_to_power_of_2=True,
        )
        grad_output_scaled = grad_output.to(torch.float32) * grad_output_scales
        grad_output_fp8_row_major = to_fp8_saturated(
            grad_output_scaled, torch.float8_e4m3fn
        )

        # Compute grad_A.
        # grad_A = grad_output @ B
        # grad_A = scaled grouped mm of (M,N) @ (B,N,K) = (M,K)
        assert not _is_column_major(grad_output_fp8_row_major), (
            "grad_output must be row-major for grad_A = grad_output @ B"
        )
        assert _is_column_major(B_fp8_col_major), (
            "B must be column-major for grad_A = grad_output @ B"
        )

        # Squeeze empty dims out of scales, to comply with grouped mm API.
        # grad_output_scales shape: (M,1) or (B, M, 1)
        # B_scales shape: (E, 1, N)
        grad_output_scales = grad_output_scales.squeeze(-1)
        B_scales = B_scales.squeeze(1)
        grad_A = torch._scaled_grouped_mm(
            grad_output_fp8_row_major,
            B_fp8_col_major,
            grad_output_scales.squeeze().reciprocal(),
            B_scales.squeeze().reciprocal(),
            offs,
            out_dtype=out_dtype,
            use_fast_accum=True,
        )

        # Convert transpose of grad_output to float8, row-major for left operand of grouped GEMM
        # needed for grad_B: grad_output_t @ A
        grad_output_t_row_major = grad_output.transpose(-2, -1).contiguous()

        # Convert A to float8, column-major for right operand of grouped GEMM:
        # needed for grad_B: grad_output @ A
        A_col_major = A.transpose(-2, -1).contiguous().transpose(-2, -1)

        # grad_B is a special case. both operands of the grouped gemm will be 2D with offsets determing the "groups."
        # Compute scales for grad_output_t and A, which are both 2D tensors with offsets which define the "jagged" groups.
        grad_output_t_fp8_row_major, grad_output_t_scales = (
            triton_fp8_row_major_jagged_rowwise_scales(
                grad_output_t_row_major,
                offs,
                torch.float8_e4m3fn,
                round_scales_to_power_of_2=True,
            )
        )

        A_fp8_col_major, A_scales = triton_fp8_col_major_jagged_colwise_scales(
            A_col_major,
            offs,
            torch.float8_e4m3fn,
            round_scales_to_power_of_2=True,
        )

        # Compute grad_B = grad_output_t @ A.
        # grad_B = grad_output_t @ A
        # grad_B = (N,M) @ (M,K) = (N,K)
        assert not _is_column_major(grad_output_t_fp8_row_major), (
            "grad_output_t must be row-major for grad_B = grad_output_t @ A"
        )
        assert _is_column_major(A_fp8_col_major), (
            "A must be column-major for grad_B = grad_output_t @ A"
        )

        # Per-token group scales computed via triton kernels above do not have
        # the empty dim like the scales computed via tensor_to_scale, so we need
        # don't need to squeeze here.
        grad_B = torch._scaled_grouped_mm(
            grad_output_t_fp8_row_major,
            A_fp8_col_major,
            grad_output_t_scales.reciprocal(),
            A_scales.reciprocal(),
            offs,
            out_dtype=out_dtype,
            use_fast_accum=True,
        )
        return grad_A, grad_B.transpose(-2, -1), None, None, None, None


class _MXFP8GroupedMM(torch.autograd.Function):
    """Differentiable implementation of grouped GEMM with dynamic mxpf8 quantization."""

    @staticmethod
    def forward(
        ctx,
        A: torch.Tensor,
        B_t: torch.Tensor,
        offs: Optional[torch.Tensor] = None,
        block_size: int = 32,
        out_dtype: Optional[torch.dtype] = torch.bfloat16,
        emulated: bool = True,
    ) -> torch.Tensor:
        # torchao _scaled_grouped_mm only supports A=2D and B=3D.
        assert A.ndim == 2, "A must be 2D"
        assert B_t.ndim == 3, "B must be 3D"
        assert block_size == 32, "Only block_size=32 is supported"
        assert emulated, "Only emulated mxfp8 grouped gemm is supported"

        # Cast to mxpf8 across dim -1.
        # A_mx shape: (M, K)
        # A_scale shape: (M, K//block_size)
        A_scale, A_mx = to_mx(A, elem_dtype=torch.float8_e4m3fn, block_size=block_size)

        # Cast B_t per-expert to mxfp8 across dim1.
        # B_t_mx shape: (E, K, N)
        # B_t_scale shape: (E, K//block_size, N)
        B_t_scale, B_t_mx = _to_mxfp8_3d_expert_weights_dim1(B_t, block_size=block_size)

        # Store what we need for backward.
        ctx.save_for_backward(A, B_t, offs)
        ctx.out_dtype = out_dtype

        # Perform scaled grouped GEMM and return result.
        # output = input @ weight.T
        # output shape: (M, N)
        out = emulated_mxfp8_scaled_grouped_mm(
            A_mx,
            A_scale,
            B_t_mx,
            B_t_scale,
            offs=offs,
            block_size=block_size,
            out_dtype=out_dtype,
        )
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        raise NotImplementedError


def _to_mxfp8_3d_expert_weights_dim1(
    w_t: torch.Tensor,  # (num_experts, K, N)
    block_size: int = 32,
    elem_dtype: torch.dtype = torch.float8_e4m3fn,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a 3D tensor of shape (experts, K, N) to MXFP8 format along dim1.
    Args:
        x (torch.Tensor): Input tensor to be converted.
        block_size (int): Block size for MXFP8 quantization.
        elem_dtype (torch.dtype): Element dtype for MXFP8 quantization.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Converted tensor and scale tensor.
            - scale shape: (expets, K // block_size, N)
            - output shape: (experts, K, N)
    """
    # To cast B_t per-expert to mxfp8 across dim1, we transpose the experts, cast along dim -1, then untranspose.
    w_scale, w_mx = to_mx(
        w_t.transpose(-2, -1).contiguous(), elem_dtype=elem_dtype, block_size=block_size
    )
    w_t_scale, w_t_mx = w_scale.transpose(-2, -1), w_mx.transpose(-2, -1)
    return w_t_scale, w_t_mx


def emulated_mxfp8_scaled_grouped_mm(
    A_mx: torch.Tensor,
    A_scale: torch.Tensor,
    B_t_mx: torch.Tensor,
    B_t_scale: torch.Tensor,
    offs: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = torch.bfloat16,
    block_size: int = 32,
) -> torch.Tensor:
    # Dequantize input
    # A_mx shape: (M, K)
    # A_scale shape: (M, K//block_size)
    A_orig_shape = A_mx.shape

    # Reshape to be able to do per-scaling group multiplication
    # A_mx shape: (M, K//block_size, block_size)
    # A_scale shape: (M, K//block_size, 1)
    A_mx = A_mx.reshape(*A_mx.shape[:-1], A_mx.shape[-1] // block_size, block_size)
    A_scale = A_scale.unsqueeze(-1)

    # Rescale and cast to bfloat16
    A = A_mx.to(torch.bfloat16) * A_scale.to(torch.bfloat16)

    # Reshape back to original shape
    # A shape: (M, K)
    A = A.reshape(A_orig_shape)

    # Dequantize weights
    # B_t_mx shape: (E, K, N)
    # B_t_scale shape: (E, K//block_size, N)
    E, K, N = B_t_mx.shape

    # Tranpose to get block_size on rightmost dim
    # B_mx shape: (E, N, K)
    # B_scale shape: (E, N, K//block_size)
    B_mx, B_scale = B_t_mx.transpose(-2, -1), B_t_scale.transpose(-2, -1)

    # Reshape to be able to do per-scaling group multiplication
    # B_mx shape: (E, N, K//block_size, block_size)
    # B_scale shape: (E, N, K//block_size, 1)
    B_mx = B_mx.reshape(*B_mx.shape[:-1], B_mx.shape[-1] // block_size, block_size)
    B_scale = B_scale.unsqueeze(-1)

    # Rescale and cast to bfloat16
    B = B_mx.to(torch.bfloat16) * B_scale.to(torch.bfloat16)

    # Reshape back to original shape
    # B shape: (E, K, N)
    B_t = B.reshape(E, N, K).transpose(-2, -1)

    # Perform bf16 grouped GEMM.
    out = torch._grouped_mm(A, B_t, offs=offs, out_dtype=out_dtype)
    return out
