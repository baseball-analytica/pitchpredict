# Analysis of the xLSTM Pitch Prediction Model
## Claude's Deep Dive into the Results

This is a goldmine! Let me dig into the fascinating patterns in this data.

---

## 1. The Sawtooth Mystery (accuracy_by_position.png)

The first thing that jumps out is this wild oscillating pattern when we plot accuracy by token position:

| Bin     | Accuracy | Pattern |
|---------|----------|---------|
| 0-24    | 66.23%   |         |
| 25-49   | 64.44%   | ↓ LOW   |
| 50-74   | 68.50%   | ↑ HIGH  |
| 75-99   | 65.29%   | ↓ LOW   |
| 100-124 | 68.91%   | ↑ HIGH  |
| 125-149 | 64.72%   | ↓ LOW   |
| 150-174 | 69.08%   | ↑ HIGH  |

A perfect ~50 token sawtooth wave! Peaks at ~69%, valleys at ~64.5%.

**Q: What causes this?**

A: It's an artifact of how we binned the data. Each pitch has exactly 16 tokens in a fixed order, and each token type has wildly different accuracy:

```
PITCH_TYPE(52.9%) → SPEED(43.4%) → ... → VY0(95.2%) → ... → PLATE_POS(23%) → RESULT(66.3%)
```

When you slide 25-token bins across this 16-token structure, different bins capture different mixes of easy vs. hard tokens. The ~50 token period comes from some resonance between the bin size (25) and pitch structure (16).

**Q: Why does it dampen over time?**

A: Early in sessions, most pitchers are at similar positions, so the token-type alignment is consistent across sessions. Later, cumulative variance in plate appearance lengths (1-10+ pitches each) causes the token types to "smear" across bins, washing out the pattern.

This is actually a cautionary tale about binning sequential data with internal structure!

---

## 2. The Hierarchy of Predictability

Not all tokens are created equal. Here's the full spectrum:

| Category | Accuracy | Why? |
|----------|----------|------|
| Vel Y | 95.2% | Almost deterministic - if you know speed, you know forward velocity |
| Release Z | 78.7% | Pitcher's arm slot is consistent within a game |
| Accel Y | 77.7% | Determined by pitch type and speed |
| Spin Axis | 76.8% | Constrained by pitch type |
| Extension | 77.8% | Pitcher mechanics are consistent |
| Spin Rate | 74.6% | Varies by pitch type, but predictable |
| Vel X/Z | 72-74% | Movement profile follows from pitch type |
| Accel X/Z | 72-77% | Same story |
| Release X | 66.4% | Some variation in horizontal release |
| Result | 66.3% | Ball/strike/hit - depends on everything |
| Pitch Type | 52.9% | Strategic choice - intentionally varied |
| Speed | 43.4% | Fine-grained (40 buckets) + intentional variation |
| Plate Pos X | 23.9% | Where it crosses the plate - highest variance |
| Plate Pos Z | 23.2% | Same - location is genuinely unpredictable |

**Q: Why is VY0 so easy but Speed so hard?**

A: VY0 (forward velocity toward home plate) is basically just speed with a coordinate transformation. Once you predict "95 mph fastball," the forward velocity component is ~95% determined. But Speed itself has 40 fine-grained buckets - the model knows "it's a fastball" but struggles with "is it 94 or 96 mph?" Pitchers also intentionally vary velocity to keep batters off-balance.

**Q: Why is plate position nearly random?**

A: With 18 classes for X and 26 for Z, pure random would be ~5% and ~4%. So the model IS learning something (23% >> 5%), but not much. This makes physical sense:

1. Even pitchers with elite "command" miss their spots by inches
2. By the time you've predicted pitch type, speed, spin, and trajectory, you've used all the signal
3. Where the ball actually crosses the plate has irreducible noise

The confusion matrices confirm this - there's a weak diagonal but massive concentration in middle bins. The model learns "predict center-ish" because that minimizes expected error when you're uncertain.

---

## 3. The Model Learned Real Baseball Strategy

### Count Effects (accuracy_by_count.png)

| Count | Accuracy |
|-------|----------|
| 0-0   | 65.2%    |
| 3-0   | **71.0%** |
| 3-2   | 69.4%    |
| 0-2   | 66.7%    |

When the pitcher is behind in the count, they become MORE predictable!

**Q: Why is 3-0 the most predictable count?**

A: At 3-0, the pitcher is one ball away from a walk. They MUST throw a strike. Most likely: fastball down the middle. Limited options = lower entropy = easier prediction.

At 0-2, the opposite - pitcher has freedom to "waste" a pitch, throw anything anywhere. More options = higher entropy = harder prediction.

This is beautiful confirmation the model learned real baseball strategy, not just token statistics.

### Pitch Transitions (pitch_transitions.png)

The transition matrix reveals that pitchers repeat themselves - strong diagonal dominance. Notable patterns:

- **KN → KN**: Knuckleballers throw almost exclusively knuckleballs (they're specialists)
- **IN → IN**: Intentional ball sequences stay as intentional balls
- **FF as universal follow-up**: From almost any pitch, fastball is a common next choice
- **Breaking ball clusters**: Curveballs, sliders, sweepers often follow each other (pitcher "working the breaking stuff")

---

## 4. Calibration: The Model Knows What It Doesn't Know

Mean Confidence: 67.12%
Mean Accuracy: 65.81%
ECE (Expected Calibration Error): 0.0131

The model is only 1.3 percentage points overconfident on average. This is remarkably good - most neural networks are far more overconfident.

The reliability diagram shows near-perfect calibration: when the model says "40% confident," it's right about 40% of the time. When it says "80% confident," it's right about 80% of the time.

**Q: Why does calibration matter?**

A: A well-calibrated model is actually *useful*. You can trust its uncertainty estimates. If it says "I'm very unsure about this pitch location" (low confidence) and "I'm quite sure this is a fastball" (high confidence), you can believe it - and make decisions accordingly.

---

## 5. How Good Is The Model Really? (Baseline Comparison)

Let's put the 65.8% accuracy in context:

| Method | Accuracy | Description |
|--------|----------|-------------|
| Uniform Random | 8.69% | Random guess from valid next tokens |
| Most Frequent | 28.54% | Always predict the mode of each category |
| **xLSTM Model** | **65.83%** | Our model |

The model captures:
- 62.6% of the gap between random and perfect
- 52.2% of the gap between most-frequent and perfect

**Q: Which categories benefit most from the model?**

| Category | Random | MostFreq | Model | Improvement |
|----------|--------|----------|-------|-------------|
| VY0 | 14.3% | 37.1% | 95.2% | +58.2 pp |
| Spin Axis | 8.3% | 26.6% | 76.8% | +50.3 pp |
| Speed | 2.3% | 8.7% | 43.4% | +34.7 pp |
| Pitch Type | 4.8% | 34.6% | 52.9% | +18.3 pp |
| Plate Pos | ~5% | ~11% | ~23% | +12 pp |

VY0 and Spin Axis show massive gains because they're highly constrained by earlier tokens in the sequence. Plate position shows the smallest gain because it genuinely has high irreducible variance - the model can't predict what isn't predictable.

---

## 6. The First Pitch Phenomenon

To eliminate the sawtooth artifact, we analyzed accuracy by actual pitch number (1st pitch, 2nd pitch, etc.) within each session.

### Overall Accuracy by Pitch Number (accuracy_by_pitch_number.png)

| Pitch # | Accuracy | Samples  |
|---------|----------|----------|
| 1       | 62.59%   | 66,688   |
| 2       | 64.68%   | 66,320   |
| 4       | 65.93%   | 65,360   |
| 10      | 66.07%   | 57,936   |
| 50      | 65.48%   | 14,976   |

Overall accuracy climbs from 62.6% to ~66% by pitch 4, then plateaus. Naive interpretation: "the model needs ~4 pitches to learn the pitcher."

**But this is misleading!** The overall number masks opposing trends at the category level...

### The Paradox: Opposite Trends by Category (category_pitch_heatmap.png)

**CATEGORIES THAT GET HARDER:**

| Category   | Pitch 1 | Pitch 5 | Pitch 50 |
|------------|---------|---------|----------|
| Pitch Type | 66.7%   | 56.8%   | 46.5%    |
| Result     | 70.5%   | 65.2%   | 64.7%    |

**CATEGORIES THAT GET EASIER:**

| Category   | Pitch 1 | Pitch 5 | Pitch 50 |
|------------|---------|---------|----------|
| Speed      | 34.0%   | 44.0%   | 42.9%    |
| Release X  | 49.5%   | 67.0%   | 66.9%    |
| Release Z  | 64.0%   | 78.4%   | 81.9%    |

**CATEGORIES THAT STAY FLAT:**

| Category   | Pitch 1 | Pitch 50 |
|------------|---------|----------|
| Vel Y      | 95.1%   | 94.6%    |
| Plate X/Z  | ~23%    | ~23%     |

**Q: Wait, pitch type accuracy DROPS from 67% to 47%? What's going on?**

A: The first pitch is special! Look at the pitch type distribution:

```
PITCH 1:                    PITCH 5+:
  FF (4-seam)    45.3%        FF (4-seam)    33.9%
  SI (Sinker)    22.9%        SI (Sinker)    16.6%
  SL (Slider)    14.1%        SL (Slider)    15.7%
  CH (Changeup)   2.2%        CH (Changeup)  11.0%  <- 5x increase!
```

68% of first pitches are fastballs vs only 50% later. Entropy increases from 2.29 bits to 2.75 bits (20% more uncertainty).

The model correctly learns: "First pitch? Probably fastball." This is real baseball - pitchers establish the strike zone early with fastballs, then mix in offspeed as the at-bat develops.

**Q: Why do physical attributes get easier?**

A: The categories that improve (Speed, Release Point, Spin, Acceleration) are all characteristics of the pitcher's delivery mechanics. By pitch 4-5, the model has seen enough of THIS pitcher to learn:

- "This pitcher releases from position X"
- "This pitcher throws around Y mph"
- "This pitcher's arm slot is at height Z"

Pitcher mechanics are consistent within a game - so more examples help.

**Q: How do we resolve the paradox - pitch type gets harder but overall goes up?**

A: The physical attributes (14 categories) outnumber the strategic ones (Pitch Type, Result = 2 categories). Most categories improve with context, which more than compensates for Pitch Type getting harder.

The model learns two fundamentally different things:
1. **PHYSICAL patterns** (release point, velocity, spin) - improves with more examples
2. **STRATEGIC patterns** (pitch selection) - gets harder as pitchers intentionally mix more

This is a beautiful separation: mechanics are learnable, but pitch selection is *intentionally* unpredictable. The model correctly identifies both.

---

## 7. Key Takeaways

| Finding | Insight |
|---------|---------|
| Sawtooth in position accuracy | Artifact of 16-token pitch structure vs bin size - be careful with binning! |
| Plate position ~23% | Location is genuinely unpredictable - highest physical variance |
| VY0 at 95% | Forward velocity is determined by speed (already in context) |
| 3-0 count → 71% accuracy | Pitchers behind in count become predictable - real strategy! |
| Speed harder than pitch type | Fine-grained buckets (40) + intentional variation |
| Excellent calibration (ECE=0.013) | Model knows its uncertainty |
| Pitch type: 67% → 47% over session | First pitch fastball phenomenon - strategic entropy increases |
| Release point: 50% → 67% over session | Model learns pitcher mechanics from examples |
| 65.8% overall vs 28.5% most-frequent | +37 pp improvement - real learning, not just memorizing frequencies |

The first pitch of a session is special: **strategically predictable** (fastball) but **physically unknown** (haven't seen this pitcher yet). As the game progresses, strategy becomes unpredictable while mechanics become learnable. The model captures both dynamics correctly.

This is a genuinely successful model that learned real baseball physics and strategy!
