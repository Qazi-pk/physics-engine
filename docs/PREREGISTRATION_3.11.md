# PIR 3.11 pre-registration (DRAFT: not yet public; becomes binding when committed and pushed to main before any 3.11 benchmark run)

Method as for 3.6-3.10: rules fixed before the runs; development half = 60 laws and blind half = 59 laws, split by SHA-256 of the dataset name (odd = blind); the blind half is counted once and never studied; Strogatz-14 is never used for design (report only); fixed scorer (#231).

## Changes under test (all behind `composition=True`, default False)
- **W1** Readable exact forms. Applied only when the final model is exact. A rewrite must evaluate identically on the training rows (max |diff| <= 1e-9 x max|y|) and have no more operations than the original. Output form chosen so the scorer can verify it (SRBench issue #235, disclosed).
- **W2** Composition family y = C*M(A)*g(k*u): u a monomial (exponents -2,-1,1,2, at most 3 variables per term, positive on all rows), g from the fixed library in `vcomp.LIB`. Exact use (noise 0): max error < 1e-6, fewest operations; only if the current model is not exact and n >= 2,000.
- **W3** Noisy acceptance as V: held-out MSE <= 1.02 x unsnapped and <= 1.05 x the best flexible reference; replacement only if held-out MSE is lower than the current model's.
- **W4** Held-out MSE <= 0.5 x var(y_val).
- **W5** Composition replaces another accepted family only if its held-out MSE < 0.98 x theirs.
- **V9** No new stage below 2,000 training rows.

## Pass rules (all must hold, else composition stays off in the release config)
1. Development half, both seeds (23654, 78901), noise 0 / 0.001 / 0.01 / 0.1: **0 laws lost** versus 3.10.0.
2. Black-box set: no dataset worse than 3.10.0 by more than 0.01 R2.
3. Gains and losses reported separately for the development and blind halves; blind half run once.
4. Unit tests (53 at the time of writing) pass; with the flag off, outputs are identical to 3.10.0.

## Not part of this pre-registration
Dimensionless-group reduction (item 3), more rows for V (item 4), test_12 work budget, Strogatz fixed-scorer rescoring: separate pre-registrations if built.

## Open before publishing
- Whether composition returns nothing when candidates are not separable at high noise (seen at 0.1 on III_4_32, I_30_5, I_48_2). Rule to be decided from the development-half real-file check, then fixed here.
