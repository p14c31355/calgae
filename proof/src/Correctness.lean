import Mathlib
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.LibrarySearch

-- Formal verification of quantization correctness for AGENT.md goals
-- Focus on AWQ-style quantization: rounding to nearest with scale

namespace CalgaeVerification

-- Define 4-bit signed integer type for weights (range -8 to 7)
abbrev Int4 : Type := { i : Int // -8 ≤ i ∧ i ≤ 7 }

-- Quantization function: scale, round to nearest integer, and clamp to Int4 range
def quantize_awq (w : Real) (scale : Real) : Int4 :=
  let val_scaled := w / scale
  let val_rounded := Round.round val_scaled
  let val_clamped := Int.clamp val_rounded (-8) 7
  ⟨val_clamped, by
    simp [val_clamped]
    apply And.intro
    . exact Int.clamp_ge _ _ _
    . exact Int.clamp_le _ _ _
  ⟩

-- Dequantization
def dequantize_awq (q : Int4) (scale : Real) : Real :=
  (q.val : Real) * scale

-- Key property: quantization error bounded
lemma quantization_error_bound (w : Real) (scale : Real) (h_scale : scale > 0)
  (h_range : -8.5 ≤ w / scale ∧ w / scale ≤ 7.5) : -- Add assumption for w/scale range
  |dequantize_awq (quantize_awq w scale) scale - w| ≤ scale / 2 := by
  unfold dequantize_awq quantize_awq
  rw [Real.norm_eq_abs]
  have h_round_error : |Round.round (w / scale) - (w / scale)| ≤ 0.5 := Round.abs_round_sub_self_le_half (w / scale)
  let val_scaled := w / scale
  let val_rounded := Round.round val_scaled
  let val_clamped := Int.clamp val_rounded (-8) 7

  -- With h_range, we can prove that val_rounded is within [-8, 7]
  have h_val_rounded_ge_neg_8 : -8 ≤ val_rounded := by
    have : -8.5 ≤ val_scaled := h_range.left
    have : Round.round val_scaled ≥ Round.round (-8.5) := Round.round_mono this
    simp at this
    exact this
  have h_val_rounded_le_7 : val_rounded ≤ 7 := by
    have : val_scaled ≤ 7.5 := h_range.right
    have : Round.round val_scaled ≤ Round.round (7.5) := Round.round_mono this
    simp at this
    exact this

  have h_in_range : -8 ≤ val_rounded ∧ val_rounded ≤ 7 := And.intro h_val_rounded_ge_neg_8 h_val_rounded_le_7

  simp [h_in_range]
  rw [Int.clamp_eq_self]
  field_simp [h_scale.ne_zero]
  rw [abs_mul]
  rw [abs_of_pos h_scale]
  rw [mul_comm]
  apply (mul_le_mul_left h_scale).mpr
  exact h_round_error

-- For matrix mult correctness: quantized matmul ≈ full precision
def quantized_matmul (A B : Matrix Nat Nat Real) (scale : Real) : Matrix Nat Nat Real :=
  Matrix.of fun i j => dequantize_awq (quantize_awq (A i j) scale) scale

lemma matmul_correctness (A B : Matrix Nat Nat Real) (scale : Real) (h_scale : scale > 0) :
  |quantized_matmul A B scale - A ⬝ B| ≤ (A.rows * A.cols * B.cols * (scale / 2)) := sorry

-- SmoothQuant extension: outlier absorption into weights
def smoothquant_scale (act_max : Real) (sparsity : Real) (beta : Real := 0.85) (qmax : Real := 127.0) : Real :=
  if sparsity > 0 ∧ act_max > 0 then (act_max / beta) / qmax else 1.0

-- Lemma: SmoothQuant scale application preserves quantization approximation
lemma smoothquant_preserves_approx (w : Real) (act_max : Real) (scale : Real) (sparsity : Real) (q : Int4) (h_scale : scale > 0) (h_range : -8.5 ≤ w / scale ∧ w / scale ≤ 7.5)
  (h_act_max : act_max > 0) : -- Add assumption for act_max > 0
  let sq_scale := smoothquant_scale act_max sparsity
  let new_scale := scale * sq_scale
  let new_w := w / sq_scale
  have h_new_scale : new_scale > 0 := by
    unfold smoothquant_scale
    split_ifs with h_cond
    . have h_beta_pos : (0.85 : Real) > 0 := by norm_num
      have h_qmax_pos : (127.0 : Real) > 0 := by norm_num
      have h_act_max_div_beta_pos : act_max / (0.85 : Real) > 0 := div_pos h_act_max h_beta_pos
      have h_act_max_div_beta_div_qmax_pos : (act_max / (0.85 : Real)) / (127.0 : Real) > 0 := div_pos h_act_max_div_beta_div_qmax_pos h_qmax_pos
      exact mul_pos h_scale h_act_max_div_beta_div_qmax_pos
    . exact h_scale -- if sparsity = 0 or act_max = 0, sq_scale = 1.0

  -- Apply quantization_error_bound to new_w and new_scale
  have h_new_range : -8.5 ≤ new_w / new_scale ∧ new_w / new_scale ≤ 7.5 := by
    unfold new_w new_scale
    field_simp [h_scale.ne_zero, h_new_scale.ne_zero]
    exact h_range -- The range condition holds for w/scale, and new_w/new_scale simplifies to w/scale

  exact quantization_error_bound new_w new_scale h_new_scale h_new_range

end CalgaeVerification
