<!---
This file is used to generate your project datasheet. Please fill in the information below and delete any unused
sections.

You can also include images in this folder and reference them in the markdown. Each image must be less than
512 kb in size, and the combined size of all images must be less than 1 MB.
-->

## How it works

A tiny matrix-multiply accelerator: a 2x2 **output-stationary systolic array** of four signed int8
multiply-accumulate (MAC) cells, the same structure used (at vastly larger scale) in TPU-style AI chips.

- Operands: two 2x2 matrices **A** and **B** of signed 8-bit integers (-128..127).
- Each cell (i, j) holds a 20-bit signed accumulator for C[i][j]. Rows of A flow right, columns of B flow
  down, one hop per clock, with the edges skewed so A[i][k] and B[k][j] meet in cell (i, j).
- One full product takes **4 clock cycles** of compute (3N-2 for N = 2). All four cells work in parallel.
- **ACC mode**: start with ACC high and C is not cleared, so C += A x B. That lets the host compute larger
  inner dimensions (a 2xK by Kx2 product) chunk by chunk. 20 bits hold 15 worst-case chunks
  (K = 30, every operand -128) without overflow; beyond that the accumulator wraps (two's complement).
- Results are read out as 3 bytes each (24-bit two's complement, sign-extended, little-endian), row-major.

This is a teaching-scale accelerator (4 MACs). It is not comparable to a GPU or a datacenter AI chip.

## Pins

| Pin | Dir | Function |
|---|---|---|
| ui[7:0] | in | Data byte to load |
| uo[7:0] | out | Current result byte |
| uio[0] | in | WR: on each clock while high, store ui_in into the next operand slot |
| uio[1] | in | START: begin a multiply (ignored while BUSY) |
| uio[2] | in | RD_NEXT: on each clock while high, advance to the next result byte (wraps after 12) |
| uio[3] | in | ACC: with START, accumulate onto C instead of clearing it |
| uio[4] | in | RST_PTR: reset the load and read pointers |
| uio[5] | out | BUSY |
| uio[6] | out | DONE (result ready) |
| uio[7] | out | LOADED (all 8 operand bytes received) |

## How to test

All strobes are sampled on the rising clock edge, so drive the pins between edges.

1. Reset (rst_n low for a few clocks).
2. Pulse RST_PTR for one clock.
3. Hold WR high for 8 clocks, presenting A[0][0], A[0][1], A[1][0], A[1][1], B[0][0], B[0][1], B[1][0], B[1][1]
   on ui_in (two's complement bytes). LOADED goes high.
4. Pulse START for one clock. BUSY is high for 4 clocks, then DONE goes high.
5. Read uo_out, pulse RD_NEXT, repeat, for 12 bytes: C[0][0] (bytes 0..2), C[0][1], C[1][0], C[1][1].

Example: A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]] gives C = [[19, 22], [43, 50]], read as
`13 00 00 16 00 00 2B 00 00 32 00 00` (hex).

The RP2040 on the Tiny Tapeout demo board can drive this directly from MicroPython.

## External hardware

None. The demo board's RP2040 (or any microcontroller) drives the pins.
