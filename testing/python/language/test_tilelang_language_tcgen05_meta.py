"""Tests for TCGEN05 metadata and instruction descriptor helpers."""

from tilelang import _ffi_api
from tvm import DataType


TF32 = DataType("custom[tfloat32]")
F32 = DataType("float32")


def _tcgen05_meta(
    m: int,
    n: int,
    k: int,
    disable_2cta: bool = False,
    disable_ws: bool = False,
) -> list[int]:
    return [int(x) for x in _ffi_api.get_tcgen5_mma_meta(m, n, k, TF32, F32, disable_2cta, disable_ws)]


def _decode_tcgen05_dtype_formats(desc: int) -> tuple[int, int, int]:
    c_format = (desc >> 4) & 0x3
    a_format = (desc >> 7) & 0x7
    b_format = (desc >> 10) & 0x7
    return c_format, a_format, b_format


def test_tcgen05_tf32_meta_selects_1cta_and_ws_shapes():
    assert _tcgen05_meta(128, 64, 8, disable_2cta=True) == [128, 64, 8, 0, 0]
    assert _tcgen05_meta(64, 64, 8, disable_2cta=True) == [64, 64, 8, 1, 0]
    assert _tcgen05_meta(64, 64, 8, disable_2cta=True, disable_ws=True) == [64, 64, 8, 0, 0]


def test_tcgen05_tf32_meta_selects_2cta_shapes():
    assert _tcgen05_meta(128, 64, 8) == [256, 64, 8, 0, 1]
    assert _tcgen05_meta(64, 64, 8) == [128, 64, 8, 0, 1]

    # TF32 2CTA supports N in multiples of 32; fall back to 1CTA for atom_n=16.
    assert _tcgen05_meta(128, 16, 8) == [128, 16, 8, 0, 0]


def test_tcgen05_tf32_meta_rejects_unsupported_shapes():
    assert _tcgen05_meta(128, 64, 4, disable_2cta=True) == []
    assert _tcgen05_meta(32, 64, 8, disable_2cta=True) == []


def test_tcgen05_tf32_instr_desc_dtype_formats():
    desc = int(_ffi_api.get_tcgen5_instr_desc(128, 64, 8, TF32, TF32, F32, True, True, 1, 1))
    c_format, a_format, b_format = _decode_tcgen05_dtype_formats(desc)

    assert c_format == 1
    assert a_format == 2
    assert b_format == 2


if __name__ == "__main__":
    test_tcgen05_tf32_meta_selects_1cta_and_ws_shapes()
    test_tcgen05_tf32_meta_selects_2cta_shapes()
    test_tcgen05_tf32_meta_rejects_unsupported_shapes()
    test_tcgen05_tf32_instr_desc_dtype_formats()
    print("ok")
