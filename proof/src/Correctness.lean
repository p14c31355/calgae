import Mathlib

def add (a b : Nat) : Nat := a + b

def fact (n : Nat) : Nat := Nat.factorial n

def add_correct (a b : Nat) : add a b = a + b := by
  simp [add]

theorem factorial_correct (n : Nat) : n.factorial = fact n := by
  induction n with
  | zero => simp [Nat.factorial, fact]
  | succ n ih => simp [Nat.factorial, fact, ih]

example (a b : Nat) : a + b = b + a := by
  apply Nat.add_comm
