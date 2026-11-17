# PIR 3.11 pre-registration (DRAFT: not yet public; becomes binding when committed and pushed to main before any 3.11 benchmark run)

Method as for 3.6-3.10: rules fixed before the runs; development half = 60 laws and blind half = 59 laws, split by SHA-256 of the dataset name (odd = blind); the blind half is counted once and never studied; Strogatz-14 is never used for design (report only); fixed scorer (#231).

## Changes under test (all behind `composition=True`, default False)
- **W1** Readable exact forms. Applied only when the final model is exact. A rewrite must evaluate identically on the training rows (max |diff| <= 1e-9 x max|y|) and have no more operations than the original. Output form chosen so the scorer can verify it (SRBench issue #235, disclosed).
  - **W1a** Separate roots are merged only when every half-integer power in the product is a square root (exponent +-1/2) of a monomial (no sum inside the root); otherwise the product is left unchanged.
  - **W1b** A perfect square under a root is written as the root of the square.
- **W2** Composition family y = C*M(A)*g(k*u): u a monomial (exponents -2,-1,1,2, at most 3 variables per term, positive on all rows), g from the fixed library in `vcomp.LIB`. Exact use (noise 0): max error < 1e-6, fewest operations; only if the current model is not exact and n >= 2,000.
- **W3** Noisy acceptance as V: held-out MSE <= 1.02 x unsnapped and <= 1.05 x the best flexible reference; replacement only if held-out MSE is lower than the current model's.
- **W4** Held-out MSE <= 0.5 x var(y_val).
- **W5** Composition replaces another accepted family only if its held-out MSE < 0.98 x theirs.
- **V9** No new stage below 2,000 training rows.

## What the development half already told us (disclosure)
The rules above were shaped by development-half results, so the development half is no longer an independent test of them: W5 (found on I_12_1 at noise 0.001 in a synthetic check) and W1a (found on test_5 at noise 0, real files). Real development-half runs with the pre-final code: 8 of 8 (noise 0 / 0.001 / 0.01 / 0.1, seeds 23654 and 78901) lost 0 laws versus 3.10.0. Seed 23654 at 0.001, 0.01 and 0.1 used code before the W1a fix. The blind half has not been run and is the independent test.

## Pass rules (all must hold, else composition stays off in the release config)
1. Development half, both seeds (23654, 78901), noise 0 / 0.001 / 0.01 / 0.1: **0 laws lost** versus 3.10.0.
2. Black-box set: no dataset worse than 3.10.0 by more than 0.01 R2.
3. Gains and losses reported separately for the development and blind halves; blind half run once.
4. Unit tests (54 at the time of writing) pass; with the flag off, outputs are identical to 3.10.0.

## Not part of this pre-registration
Dimensionless-group reduction (item 3), more rows for V (item 4), test_12 work budget, Strogatz fixed-scorer rescoring: separate pre-registrations if built.

## Decided: no extra high-noise rule
Whether composition should return nothing at high noise when candidates are not separable was an open question after synthetic checks. The real development-half runs at noise 0.1 (both seeds) lost 0 laws, so no extra rule is added. W3, W4 and W5 stay as written; the pass rules decide.

## Strogatz-14 (report only)
Run once on the frozen code with the same driver as 3.10 (strog310.sh); nothing is designed or tuned from it. With 300 training rows the noise-tolerant stage does not run (V9), so any change would come only from the exact-data composition step at noise 0. Counts come from SRBench's original scorer (the fixed scorer cannot read the Strogatz true models).

## Frozen run plan
1. Commit and push this file to main before any 3.11 benchmark run; record the commit hash.
2. Run on that exact commit: development half, both seeds, four noise levels; black-box set; blind half once; Strogatz-14 report only.
3. Report counts by half, output-form gains separately from discovery gains, and counts with and without the positivity assumption (#235) where they differ.
4. If any pass rule fails, composition stays off in the release config and the failure is reported.
