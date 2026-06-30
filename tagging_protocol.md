# Tagging Protocol — Kaenguru 11-tag taxonomy

## Tag set

```
arithmetik
algebra
zahlentheorie
ziffern
folgen
logik
kombinatorik
geometrie_2d
geometrie_3d
raeumliches_denken
winkel
```

## Guiding principle

Tags exist for **error interpretation**, not for surface description. Assign tags that identify the main reasoning methods a solver must use, or where a model is likely to fail.

- Prefer 1–2 tags. Use 3 only when three distinct failure modes are central. 4+ is rare.
- Order tags from most central to most auxiliary.
- A tag is justified iff removing it would make a future error analysis less informative.

## Per-tag criteria

### `arithmetik`
Concrete numerical computation: basic operations, fractions, ratios, percentages, averages, totals, unit conversions, estimation, rounding, speed/time/distance solved by direct calculation.
- Don't tag when arithmetic is only the trivial final step after another method dominates.

### `algebra`
**Symbolic variables / parameters / unknowns are central.** Equations with letters, systems, functional equations `f(x+y) = f(x)f(y)`, polynomial identities, parameterized expressions, symbolic optimization, generic-N proofs.
- An equation written purely with numbers is **arithmetik**, not algebra.
- A single unknown "solved" by direct subtraction or division is **arithmetik**.
- Use both `algebra` + `arithmetik` only when both genuinely co-dominate.

### `zahlentheorie`
Integer structure central: divisibility, primes, factors/multiples, parity, remainders/modular reasoning, gcd/lcm, factorial prime exponents.
- Not for "the answer happens to be an integer".

### `ziffern`
Decimal representation is the object: digits, Quersumme, palindromes, reversed digits, concatenated number cards, digit products, symbolic-digit notation `abcd`, decimal-comma insertion.
- Pair with `zahlentheorie` when both digits and divisibility matter (e.g. divisibility by 9 via Quersumme).

### `folgen`
Ordered progression or repeated structure: sequences, periodic patterns, indexed positions, spirals, repeated/iterative procedures, "after n steps", cyclic movement, repeated jumps.
- Not for "multi-step solution".

### `logik`
Non-computational constraint reasoning: elimination, contradiction, invariants, truth/lie, ordering/scheduling constraints, strategy, "must be true / cannot happen", logical case analysis.
- Not a generic "requires thinking" tag.

### `kombinatorik`
Counting possibilities is central: arrangements, permutations, paths, colorings, distributions, schedules, probability (favorable vs. total cases).
- Not for small case-checking unless the answer itself is a count of configurations.

### `geometrie_2d`
Planar geometry — **including text-described figures even without a diagram**: lengths, areas, perimeters, triangles/polygons/circles, planar tilings, coordinate geometry, planar Pythagoras, 2D symmetry, grid/board figures, paths in a plane.
- Be generous: any problem whose object of reasoning is a 2D figure gets this tag, even if the answer comes via arithmetic.

### `geometrie_3d`
Solid/3D geometry: cubes, cuboids, tetrahedra, pyramids, cylinders, spheres, volumes, surface areas, nets, 3D paths, cross-sections, 3D block arrangements.

### `raeumliches_denken` (add-on)
**Refines** a geometry domain tag when mental visualization is a major source of difficulty: mental rotation, folding, rolling, dice/cube orientation, hidden/opposite faces, interpreting nets, transparent overlays, views from different directions, tracking position/orientation.
- Does not replace `geometrie_2d` / `geometrie_3d`.
- Not "a diagram exists".

### `winkel` (add-on)
**Refines** a geometry domain tag when angle reasoning drives the solution: angle chasing, angle sums, parallel-line relations, rotations by a specified angle, central/inscribed angles, cyclic angle arguments.
- Not "a triangle appears".

## Interaction rules

- Every geometric problem gets `geometrie_2d` OR `geometrie_3d`; add `raeumliches_denken` / `winkel` only as refinements.
- `arithmetik` vs `algebra`: concrete numbers → arithmetik; symbolic variables → algebra.
- `ziffern` vs `zahlentheorie`: digits as objects → ziffern; divisibility/primes → zahlentheorie; pair when both apply.
- `folgen` vs `kombinatorik`: ordered progression → folgen; counting possibilities → kombinatorik; pair for "how many sequences reach the end".
- `logik` pairs with other tags when constraint reasoning is the actual method, not just present.

## When to omit a tag

- No `arithmetik` for trivial final calculation after another method.
- No `algebra` for a direct numeric computation, even if the problem mentions an unknown.
- No `logik` just because reasoning is involved.
- No `kombinatorik` for small case-checking unless counting is the goal.
- No `winkel` just because a triangle appears.
- No `raeumliches_denken` just because there is a diagram.
- No `zahlentheorie` just because numbers are integers.

## Example output

```json
"tags": ["geometrie_3d", "raeumliches_denken"]
"tags": ["ziffern", "zahlentheorie"]
"tags": ["kombinatorik", "logik"]
"tags": ["geometrie_2d", "winkel", "kombinatorik"]
"tags": ["algebra"]
"tags": ["arithmetik", "prozent"]   // INVALID — prozent not in tag set; use arithmetik only
```
