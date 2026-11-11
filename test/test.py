# SPDX-FileCopyrightText: © 2026 Sumit Jindal
# SPDX-License-Identifier: Apache-2.0
#
# cocotb tests for tt_um_sumit_int8_systolic. Every result is checked against a
# pure-Python golden model. Works with cocotb 1.9.x and 2.x.

import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, RisingEdge

N = 2
ACC_W = 20
STEPS = 3 * N - 2
N_RANDOM = int(os.environ.get("N_RANDOM", "1000"))

WR, START, RD, ACC, RSTP = 1, 2, 4, 8, 16
BUSY, DONE, LOADED = 0x20, 0x40, 0x80


# ---------------- golden model ----------------
def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(N)) for j in range(N)] for i in range(N)]


def wrap(v, bits=ACC_W):
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) else v


def u8(v):
    return v & 0xFF


def s24(b0, b1, b2):
    v = b0 | (b1 << 8) | (b2 << 16)
    return v - (1 << 24) if v & 0x800000 else v


def rand_mat(rng, lo=-128, hi=127):
    return [[rng.randint(lo, hi) for _ in range(N)] for _ in range(N)]


# ---------------- driver helpers ----------------
async def setup(dut):
    clock = Clock(dut.clk, 10, "us")  # 100 kHz, positional unit works on cocotb 1.9 and 2.x
    cocotb.start_soon(clock.start())
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    await FallingEdge(dut.clk)
    dut.rst_n.value = 1
    await FallingEdge(dut.clk)


async def pulse(dut, bits, data=0):
    """Drive for exactly one rising edge (inputs change on falling edges)."""
    dut.ui_in.value = data
    dut.uio_in.value = bits
    await FallingEdge(dut.clk)
    dut.uio_in.value = 0
    dut.ui_in.value = 0


async def load(dut, a, b):
    await pulse(dut, RSTP)
    for m in (a, b):
        for i in range(N):
            for j in range(N):
                await pulse(dut, WR, u8(m[i][j]))


async def run(dut, acc=False):
    await pulse(dut, START | (ACC if acc else 0))
    cycles = 0
    while int(dut.uio_out.value) & BUSY:
        await FallingEdge(dut.clk)
        cycles += 1
        assert cycles < 100, "BUSY never cleared"
    assert int(dut.uio_out.value) & DONE, "DONE not set after run"
    return cycles


async def read_all(dut):
    c = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            bs = []
            for _ in range(3):
                bs.append(int(dut.uo_out.value))
                await pulse(dut, RD)
            c[i][j] = s24(*bs)
    return c


async def matmul_on_chip(dut, a, b, acc=False):
    await load(dut, a, b)
    await run(dut, acc)
    return await read_all(dut)


def check(got, want, ctx=""):
    assert got == want, f"{ctx} got {got} want {want}"


# ---------------- tests ----------------
@cocotb.test()
async def test_known_values(dut):
    await setup(dut)
    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    check(await matmul_on_chip(dut, a, b), [[19, 22], [43, 50]], "2x2 example")
    ident = [[1, 0], [0, 1]]
    m = [[-7, 99], [127, -128]]
    check(await matmul_on_chip(dut, ident, m), m, "I*M")
    check(await matmul_on_chip(dut, m, ident), m, "M*I")
    zero = [[0, 0], [0, 0]]
    check(await matmul_on_chip(dut, m, zero), zero, "M*0")


@cocotb.test()
async def test_extremes(dut):
    await setup(dut)
    lo = [[-128] * N for _ in range(N)]
    hi = [[127] * N for _ in range(N)]
    check(await matmul_on_chip(dut, lo, lo), matmul(lo, lo), "-128*-128")  # 32768 per element
    check(await matmul_on_chip(dut, lo, hi), matmul(lo, hi), "-128*127")
    check(await matmul_on_chip(dut, hi, hi), matmul(hi, hi), "127*127")


@cocotb.test()
async def test_random(dut):
    await setup(dut)
    rng = random.Random(int(os.environ.get("SEED", "20261130")))
    for n in range(N_RANDOM):
        a, b = rand_mat(rng), rand_mat(rng)
        check(await matmul_on_chip(dut, a, b), matmul(a, b), f"random #{n} A={a} B={b}")
    dut._log.info(f"{N_RANDOM} random int8 matrix products matched the golden model")


@cocotb.test()
async def test_accumulate_tiles(dut):
    """ACC mode: C = sum of chunk products, i.e. a (2 x 2K) * (2K x 2) product done in chunks."""
    await setup(dut)
    rng = random.Random(7)
    for trial in range(20):
        chunks = rng.randint(2, 16)
        total = [[0] * N for _ in range(N)]
        for c in range(chunks):
            a, b = rand_mat(rng), rand_mat(rng)
            p = matmul(a, b)
            total = [[total[i][j] + p[i][j] for j in range(N)] for i in range(N)]
            await load(dut, a, b)
            await run(dut, acc=(c > 0))
        want = [[wrap(v) for v in row] for row in total]
        check(await read_all(dut), want, f"accumulate trial {trial} ({chunks} chunks)")


@cocotb.test()
async def test_accumulator_wraps_like_model(dut):
    """Past 2^19 the accumulator wraps (two's complement), exactly like the model says."""
    await setup(dut)
    lo = [[-128] * N for _ in range(N)]
    total = 0
    for c in range(20):  # 20 * 32768 = 655360 > 2^19
        await load(dut, lo, lo)
        await run(dut, acc=(c > 0))
        total += 2 * 128 * 128
    got = await read_all(dut)
    check(got, [[wrap(total)] * N for _ in range(N)], "wrap")


@cocotb.test()
async def test_timing_and_flags(dut):
    await setup(dut)
    assert int(dut.uio_oe.value) == 0xE0
    assert int(dut.uio_out.value) & (BUSY | DONE | LOADED) == 0, "flags clear after reset"
    a, b = [[1, 1], [1, 1]], [[2, 2], [2, 2]]
    await load(dut, a, b)
    assert int(dut.uio_out.value) & LOADED, "LOADED after 8 bytes"
    cycles = await run(dut)
    assert cycles == STEPS, f"BUSY lasted {cycles} cycles, expected {STEPS}"
    # readout wraps back to byte 0 after 3*N*N bytes
    first = await read_all(dut)
    again = await read_all(dut)
    check(again, first, "readout wrap")
    # writes are ignored while busy, START while busy is ignored
    await load(dut, [[3, 0], [0, 3]], [[1, 2], [3, 4]])
    await pulse(dut, START)
    await pulse(dut, WR, 0x55)       # ignored
    await pulse(dut, START | ACC)    # ignored
    while int(dut.uio_out.value) & BUSY:
        await FallingEdge(dut.clk)
    check(await read_all(dut), [[3, 6], [9, 12]], "busy ignores WR/START")


@cocotb.test()
async def test_reset_mid_run(dut):
    await setup(dut)
    await load(dut, [[9, 9], [9, 9]], [[9, 9], [9, 9]])
    await pulse(dut, START)
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 2)
    await FallingEdge(dut.clk)
    dut.rst_n.value = 1
    await FallingEdge(dut.clk)
    assert int(dut.uio_out.value) & (BUSY | DONE | LOADED) == 0
    check(await read_all(dut), [[0, 0], [0, 0]], "acc cleared by reset")
    a, b = [[2, -3], [4, 5]], [[-6, 7], [8, -9]]
    check(await matmul_on_chip(dut, a, b), matmul(a, b), "after reset")
