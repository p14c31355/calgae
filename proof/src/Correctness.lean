import Mathlib

def add (a b : Nat) : Nat := a + b

def fact (n : Nat) : Nat := Nat.factorial n

theorem add_correct (a b : Nat) : add a b = a + b := by
  simp [add]

theorem factorial_correct (n : Nat) : n.factorial = fact n := by
  simp [fact]

example (a b : Nat) : a + b = b + a := by
  apply Nat.add_comm
