/*
 * Copyright (c) 2026 Sumit Jindal
 * SPDX-License-Identifier: Apache-2.0
 *
 * tt_um_sumit_int8_systolic: a tiny int8 matrix-multiply engine.
 *
 * Computes C = A x B (or C += A x B) for N x N signed int8 matrices on an
 * output-stationary N x N systolic array of multiply-accumulate cells.
 * Results are signed ACC_W-bit accumulators, read out as 3 bytes each
 * (24-bit two's complement, little-endian).
 *
 * Host protocol (all strobes are sampled on the rising clock edge):
 *   1. Pulse RST_PTR (optional: start of a new load).
 *   2. Hold WR high for 2*N*N clocks while presenting bytes on ui_in:
 *      A[0][0], A[0][1], ... A[N-1][N-1], then B[0][0] ... B[N-1][N-1].
 *   3. Pulse START (with ACC high to add onto the previous C instead of clearing).
 *      BUSY goes high for 3N-2 clocks, then DONE goes high.
 *   4. uo_out shows result byte 0 (C[0][0] bits 7:0). Each clock with RD_NEXT
 *      high advances to the next byte: C[0][0] b0,b1,b2, C[0][1] b0..., row-major.
 */

`default_nettype none

module tt_um_sumit_int8_systolic #(
    parameter N     = 2,   // array is N x N
    parameter ACC_W = 20   // accumulator width (17 bits per pass + headroom for ACC mode)
) (
    input  wire [7:0] ui_in,    // Dedicated inputs: data byte
    output wire [7:0] uo_out,   // Dedicated outputs: result byte
    input  wire [7:0] uio_in,   // IOs: Input path
    output wire [7:0] uio_out,  // IOs: Output path
    output wire [7:0] uio_oe,   // IOs: Enable path (active high: 0=input, 1=output)
    input  wire       ena,      // always 1 when the design is powered, so you can ignore it
    input  wire       clk,      // clock
    input  wire       rst_n     // reset_n - low to reset
);

  localparam NN     = N * N;
  localparam NIN    = 2 * NN;          // bytes to load
  localparam NOUT   = 3 * NN;          // bytes to read
  localparam STEPS  = 3 * N - 2;       // K + 2(N-1) cycles, K = N
  localparam WP_W   = $clog2(NIN);
  localparam RP_W   = $clog2(NOUT);
  localparam T_W    = $clog2(STEPS + 1);

  // control pins
  wire wr      = uio_in[0];
  wire start   = uio_in[1];
  wire rd_next = uio_in[2];
  wire acc_mode= uio_in[3];
  wire rst_ptr = uio_in[4];

  // ---------------- operand storage ----------------
  reg  [8*NN-1:0] a_mem;   // A row-major, element (i,k) at byte i*N+k
  reg  [8*NN-1:0] b_mem;   // B row-major, element (k,j) at byte k*N+j
  reg  [WP_W-1:0] wptr;
  reg             loaded;

  // ---------------- sequencing ----------------
  reg             busy;
  reg             done;
  reg  [T_W-1:0]  t;
  reg  [RP_W-1:0] rptr;

  always @(posedge clk) begin
    if (!rst_n) begin
      a_mem  <= 0;
      b_mem  <= 0;
      wptr   <= 0;
      loaded <= 1'b0;
      busy   <= 1'b0;
      done   <= 1'b0;
      t      <= 0;
      rptr   <= 0;
    end else begin
      if (rst_ptr) begin
        wptr   <= 0;
        rptr   <= 0;
        loaded <= 1'b0;
      end else if (wr && !busy) begin
        if (wptr < NN) a_mem[8*wptr +: 8] <= ui_in;
        else           b_mem[8*(wptr-NN) +: 8] <= ui_in;
        if (wptr == NIN - 1) begin
          wptr   <= 0;
          loaded <= 1'b1;
        end else begin
          wptr   <= wptr + 1'b1;
        end
      end

      if (start && !busy) begin
        busy <= 1'b1;
        done <= 1'b0;
        t    <= 0;
        rptr <= 0;
      end else if (busy) begin
        if (t == STEPS - 1) begin
          busy <= 1'b0;
          done <= 1'b1;
        end
        t <= t + 1'b1;
      end else if (rd_next && !rst_ptr) begin
        rptr <= (rptr == NOUT - 1) ? {RP_W{1'b0}} : rptr + 1'b1;
      end
    end
  end

  wire flush     = start && !busy;              // empty the pipeline registers before every run
  wire clear_acc = flush && !acc_mode;          // ...and zero C unless accumulating

  // ---------------- skewed edge feeders ----------------
  // row i receives A[i][t-i] at step t; column j receives B[t-j][j].
  wire [7:0] a_edge [0:N-1];
  wire [7:0] b_edge [0:N-1];

  genvar gi, gj;
  generate
    for (gi = 0; gi < N; gi = gi + 1) begin : g_feed
      wire signed [T_W:0] ka = $signed({1'b0, t}) - gi;
      assign a_edge[gi] = (busy && ka >= 0 && ka < N) ? a_mem[8*(gi*N + ka[T_W-1:0]) +: 8] : 8'd0;
      assign b_edge[gi] = (busy && ka >= 0 && ka < N) ? b_mem[8*(ka[T_W-1:0]*N + gi) +: 8] : 8'd0;
    end
  endgenerate

  // ---------------- N x N systolic array ----------------
  wire [7:0]       a_bus [0:N-1][0:N];    // a flowing right
  wire [7:0]       b_bus [0:N][0:N-1];    // b flowing down
  wire [ACC_W-1:0] acc   [0:N-1][0:N-1];

  generate
    for (gi = 0; gi < N; gi = gi + 1) begin : g_row
      assign a_bus[gi][0] = a_edge[gi];
      assign b_bus[0][gi] = b_edge[gi];
      for (gj = 0; gj < N; gj = gj + 1) begin : g_col
        sys_pe #(.ACC_W(ACC_W)) pe (
            .clk  (clk),
            .rst_n(rst_n),
            .en   (busy),
            .flush(flush),
            .clr  (clear_acc),
            .a_in (a_bus[gi][gj]),
            .b_in (b_bus[gi][gj]),
            .a_out(a_bus[gi][gj+1]),
            .b_out(b_bus[gi+1][gj]),
            .acc  (acc[gi][gj])
        );
      end
    end
  endgenerate

  // ---------------- readout ----------------
  wire [24*NN-1:0] c_flat;
  generate
    for (gi = 0; gi < N; gi = gi + 1) begin : g_out_r
      for (gj = 0; gj < N; gj = gj + 1) begin : g_out_c
        assign c_flat[24*(gi*N+gj) +: 24] = {{(24-ACC_W){acc[gi][gj][ACC_W-1]}}, acc[gi][gj]};
      end
    end
  endgenerate

  assign uo_out  = c_flat[8*rptr +: 8];
  assign uio_out = {loaded, done, busy, 5'b0};
  assign uio_oe  = 8'b1110_0000;

  wire _unused = &{ena, uio_in[7:5], 1'b0};

endmodule

// One multiply-accumulate cell: acc += a*b (signed int8), passes a right, b down.
module sys_pe #(
    parameter ACC_W = 20
) (
    input  wire             clk,
    input  wire             rst_n,
    input  wire             en,
    input  wire             flush,
    input  wire             clr,
    input  wire [7:0]       a_in,
    input  wire [7:0]       b_in,
    output reg  [7:0]       a_out,
    output reg  [7:0]       b_out,
    output reg  [ACC_W-1:0] acc
);
  wire signed [15:0] prod = $signed(a_in) * $signed(b_in);

  always @(posedge clk) begin
    if (!rst_n) begin
      a_out <= 8'd0;
      b_out <= 8'd0;
      acc   <= {ACC_W{1'b0}};
    end else if (flush) begin
      a_out <= 8'd0;
      b_out <= 8'd0;
      if (clr) acc <= {ACC_W{1'b0}};
    end else if (en) begin
      a_out <= a_in;
      b_out <= b_in;
      acc   <= acc + {{(ACC_W-16){prod[15]}}, prod};
    end
  end
endmodule
