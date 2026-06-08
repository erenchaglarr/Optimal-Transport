Id←=⌜˜ ↕
MP←+˝∘×⎉1‿∞ 
Diag←{𝕩×Id ≠𝕩}
t←(3‿3 ⥊ ↕ 9)

Sinkhorn ← {a‿b‿c‿eps‿iters:
  k←(-c) (⋆÷) eps
  n←⊑≢a
  m←⊑≢b
  u0←n⥊1
  v0←m⥊1
  u‿v←{un‿vn:
    un1←a÷(k MP vn)
    vn1←b÷((⍉k) MP un1)
    # lam←(⌊un1) (+÷2) (⌈ vn1)
    lam ← 1
    (un1×lam)‿(vn1÷lam)
  }⍟iters u0‿v0
  #•Show Diag u
  #•Show Diag v
  #•Show k
  #(Diag u) MP k MP (Diag v)
}
NaiveLogSH←Sinkhorn⌾(⋆⁼((↕3)⊑))

points_class_a←[1‿1, 1‿2, 2‿1, 2‿2]
points_class_b←[10‿10, 2‿3]
n‿m←(⊑≢)¨points_class_a‿points_class_b
a←n⥊1
b←(n÷m)×(m⥊1)
# c←⌜

ex_a←1‿1‿0
ex_b←0‿1‿1
ex_c←[[0, 1, 1],
      [1, 0, 1],
      [1, 1, 0]]
eps←0.1
•Show Sinkhorn ex_a‿ex_b‿ex_c‿eps‿100
•Show NaiveLogSH ex_a‿ex_b‿ex_c‿eps‿100
# •Show p
