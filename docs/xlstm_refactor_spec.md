# xLSTM Integration + API Refactor Spec

## Summary

Integrate the xLSTM inference path into the library while keeping training/data generation in `tools/`.
Extend the pitcher prediction API to support autoregressive (AR) history via `prev_pitches`, using a flat
`pa_id` field to indicate plate-appearance boundaries. Maintain compatibility with the similarity algorithm.

## Goals

- Provide a first-class xLSTM inference algorithm alongside similarity.
- Accept structured AR history in `prev_pitches` and convert it into tokens + context.
- Keep model training and dataset generation code in `tools/` (not runtime package code).
- Preserve the existing response shape and add generated `pitches` output for xLSTM.
- Avoid documenting deprecated algorithm aliases.

## Non-Goals

- Expose xLSTM training via the API.
- Implement xLSTM for batter or batted-ball prediction.
- Change similarity API semantics beyond the new pitcher request type.

## API Changes

### Request Types

`PredictPitcherRequest` keeps top-level context defaults and adds/uses `prev_pitches` for AR history.
`prev_pitches` may be an empty list for cold-start xLSTM inference.

`Pitch` (used in `prev_pitches`) is extended with a required `pa_id` and optional per-pitch overrides.

```
class Pitch(BaseModel):
    # Required pitch attributes (tokenized categories)
    pa_id: int
    pitch_type: str
    speed: float
    spin_rate: float
    spin_axis: float
    release_pos_x: float
    release_pos_z: float
    release_extension: float
    vx0: float
    vy0: float
    vz0: float
    ax: float
    ay: float
    az: float
    plate_pos_x: float
    plate_pos_z: float
    result: str

    # Optional per-pitch overrides (for xLSTM context)
    batter_id: int | None = None
    batter_age: int | None = None
    batter_hits: Literal["L", "R"] | None = None
    count_balls: int | None = None
    count_strikes: int | None = None
    outs: int | None = None
    bases_state: int | None = None
    score_bat: int | None = None
    score_fld: int | None = None
    inning: int | None = None
    pitch_number: int | None = None
    number_through_order: int | None = None
```

Top-level `PredictPitcherRequest` fields remain as defaults:

- Required: `pitcher_id`, `batter_id`, `algorithm`
- Optional defaults: `pitcher_age`, `pitcher_throws`, `batter_age`, `batter_hits`, `count_balls`,
  `count_strikes`, `outs`, `bases_state`, `score_bat`, `score_fld`, `inning`, `pitch_number`,
  `number_through_order`, `game_date`, fielder IDs, days-since, strike zone.
- `prev_pitches`: required when `algorithm` is `"xlstm"` (empty list allowed).
- `sample_size`: how many pitches to generate (xLSTM) or sample (similarity).

### Response Types

`PredictPitcherResponse` already includes `pitches` (list of `Pitch`). For xLSTM, this will be
the generated samples. For similarity, it remains a sample of similar historical pitches.

## Inference Rules

### `pa_id` handling

- `pa_id` is required for any pitch in xLSTM history (ignored for cold-start).
- `PA_START` is inserted when a new `pa_id` begins.
- `PA_END` is inserted after the last pitch in a `pa_id` group.
- `SESSION_START` and `SESSION_END` are inserted at the boundaries of the request history.

### Per-pitch context defaults

Build a per-pitch context by merging:

1) Top-level defaults from `PredictPitcherRequest`
2) Per-pitch overrides from `Pitch`
3) Inferred fields (see below)

Per-pitch overrides take priority. Defaults fill missing fields.

### Inference for missing fields

Inferable:

- `pitch_number`: set to 1..N within each `pa_id` group if missing.
- `count_balls` / `count_strikes`: inferred from `result` using standard MLB rules
  (see appendix).

Not reliably inferable (must be supplied by defaults or per-pitch overrides):

- `outs`, `bases_state`, `score_bat`, `score_fld`, `inning`, `number_through_order`
- `pitcher_throws`, `batter_hits`, `pitcher_age`, `batter_age`
- fielder IDs, days-since, `strike_zone_top`, `strike_zone_bottom`

Behavior when inference fails:

- If counts cannot be inferred and are missing, reject with 400 for xLSTM.
- If other required context fields are missing, reject with 400 and list missing fields.
    - Exception: cold-start (empty `prev_pitches`) defaults `count_balls=0` and
      `count_strikes=0` if not provided.

### Cold-start behavior (`prev_pitches=[]`)

- Treat as a single new plate appearance with no prior pitches.
- Insert `SESSION_START` + `PA_START` before generation.
- If `pa_id` is needed for output, default to `1` or accept a `pa_id` override
  in `PredictPitcherRequest` later.

## xLSTM Runtime Architecture

### New Package Layout

Create `src/pitchpredict/backend/algs/xlstm/` with:

- `model.py`: `BaseballxLSTM`, `ModelConfig`, `init_gate_biases`
- `checkpoint.py`: `load_checkpoint`, `strip_compiled_prefix`
- `encoding.py`: convert `Pitch` + context into tokens + `PackedPitchContext`
- `decoding.py`: convert tokens/logits into `Pitch` + summary stats
- `sequence.py`: build tokenized history + contexts from a `PredictPitcherRequest`
- `predictor.py`: AR loop with grammar mask + sampling strategy
- `base.py`: `XlstmAlgorithm` implementing `PitchPredictAlgorithm`

### Training/Data Generation Refactor

Move legacy training/data-generation modules out of the runtime package into `tools/`.

Keep shared tokens/grammar in runtime, but locate them under `xlstm/`:

- `src/pitchpredict/backend/algs/xlstm/tokens.py` (PitchToken + categories + grammar)
- `src/pitchpredict/backend/algs/xlstm/dataset.py` (if needed for inference)

Remove the legacy training package to avoid confusion. If required for transition,
add a temporary shim module that re-exports `xlstm/*` under the old import
paths with a deprecation warning.

### Algorithm Registration

Update `src/pitchpredict/backend/algs/__init__.py`:

- Add `xlstm` to the algorithm registry.
- Keep `similarity` unchanged.

## xLSTM Inference Flow

1) **Validate request**
   - If `algorithm` is `"xlstm"`, require `prev_pitches` (empty allowed).
   - Validate `pa_id` and `result` vocab.
   - Ensure required context fields are present or inferable.

2) **Build history tokens**
   - `SESSION_START`
   - For each `pa_id`: `PA_START` + 16 tokens per pitch + `PA_END`
   - `SESSION_END`

3) **Build per-token context**
   - For pitch tokens, repeat the merged per-pitch context (16 tokens).
   - For structural tokens (`SESSION_*`, `PA_*`), reuse the nearest pitch context.

4) **Generate next pitch**
   - Take the last token of history and generate the next 16 pitch tokens.
   - Apply grammar mask using `valid_next_tokens`.
   - Use sampling strategy: default top-k (k=5) with temperature 1.0
     (documented and configurable later).
   - Generate `sample_size` pitches using batched sampling (one batch,
     16 sequential forward passes).

5) **Decode outputs**
   - Map tokens to `Pitch` fields using the same binning scheme as training.
   - Populate `pitches` in the response.
   - Produce summary stats for `basic_*` and `detailed_*` fields by aggregating
     the generated samples.

### Generation Passes (Clarification)

- One generated pitch requires 16 sequential model steps (one per token).
- With `sample_size=N`, use batch size N so the model still runs 16 forward
  passes total (not N×16), each pass on the full batch.

### Pitch Type Probabilities

- `pitch_type_probs` should come from the **step-0 logits** (the first token
  generated after `PA_START`/`RESULT`).
- Apply the grammar mask to only `PITCH_TYPE` tokens, then softmax.
- This is not averaged across steps. Optionally include a sampled frequency
  distribution for debugging.

## Mapping Rules (Tokens <-> API Fields)

- `Pitch.speed` <-> `release_speed` bins (same ranges as training)
- `Pitch.spin_rate` <-> `release_spin_rate` bins
- `Pitch.spin_axis` <-> `spin_axis` bins
- `Pitch.release_pos_x` <-> `release_pos_x` bins
- `Pitch.release_pos_z` <-> `release_pos_z` bins
- `Pitch.release_extension` <-> `release_extension` bins
- `Pitch.vx0/vy0/vz0` <-> `vx0/vy0/vz0` bins
- `Pitch.ax/ay/az` <-> `ax/ay/az` bins
- `Pitch.plate_pos_x/plate_pos_z` <-> `plate_x/plate_z` bins
- `Pitch.result` <-> `description`/`events` mapping (see vocab)

## Result Vocabulary

`Pitch.result` must map to xLSTM result tokens. Accept only these strings:

- `ball`
- `ball_in_dirt`
- `called_strike`
- `foul`
- `foul_bunt`
- `bunt_foul_tip`
- `foul_pitchout`
- `pitchout`
- `hit_by_pitch`
- `intentional_ball`
- `hit_into_play`
- `missed_bunt`
- `foul_tip`
- `swinging_pitchout`
- `swinging_strike`
- `swinging_strike_blocked`
- `blocked_ball`
- `automatic_ball`
- `automatic_strike`

If a payload uses `events`-style strings, add a normalization layer in `encoding.py`.

## Validation & Error Handling

- If xLSTM is requested without `prev_pitches` (missing or null), return 400
  with a clear error. Empty list is valid (cold-start).
- If any required context fields are missing and cannot be inferred, return 400 with field list.
- If `pa_id` is missing or non-positive for any pitch, return 400.
- If `result` is not in the vocab, return 400 with allowed values.

## Response Aggregation (xLSTM)

Maintain the existing similarity response schema:

- `basic_pitch_data`: populated from step-0 logits and/or samples
  - `pitch_type_probs`: masked softmax of step-0 logits (pre-sampling)
  - speed/x/z mean + std: aggregated from generated samples
- `detailed_pitch_data`: aggregated from generated samples (same schema as similarity)
  - fastball/offspeed breakdown
  - percentiles for speed/x/z
- `basic_outcome_data`: aggregated from generated samples using `result` tokens
- `detailed_outcome_data`: empty `{}` unless derived fields are explicitly defined

If logits are needed, expose them via a separate optional field (see below),
not by redefining `detailed_pitch_data`.

### Optional Logits Output

If a request flag like `return_logits=true` is present:

- add `prediction_metadata["logits_topk"]` (or a dedicated `model_outputs` field)
  with top-k token logits for each of the 16 steps.
- keep the existing `basic_*`/`detailed_*` fields unchanged.

## Device Handling

- Default device: `cuda:0` if available, else `cpu`.
- Override via:
  - env: `PITCHPREDICT_DEVICE`
  - constructor arg: `XlstmAlgorithm(device="cpu")`

## Model Weights Distribution

### Default Behavior

- Host the checkpoint on Hugging Face (HF).
- Lazily download the checkpoint the first time the xLSTM algorithm is used.
- Cache the file in the user cache directory:
  - `~/.cache/pitchpredict/xlstm/` (default)

### Configuration

Support overrides via environment variables and optional constructor args:

- `PITCHPREDICT_XLSTM_REPO` (HF repo id)
- `PITCHPREDICT_XLSTM_REVISION` (tag/commit, optional)
- `PITCHPREDICT_XLSTM_FILENAME` (checkpoint filename, default `ckpt.pt`)
- `PITCHPREDICT_MODEL_DIR` (override cache directory)
- `PITCHPREDICT_XLSTM_PATH` (absolute path override, skip download)

Constructor override example:

```
PitchPredict(
    algorithms={
        "xlstm": XlstmAlgorithm(checkpoint_path="/path/to/ckpt.pt")
    }
)
```

### Implementation Notes

- Use `huggingface_hub.hf_hub_download` for caching, resuming, and pinning revisions.
- Place a simple file lock around download to avoid multi-worker races.
- If download fails (offline, auth), return a clear error with the expected local path.

## Tests

Add:

- `tests/test_xlstm_sequence_builder.py`
  - `pa_id` boundaries insert `PA_START/PA_END`
  - `pitch_number` inference per PA
  - context defaults + per-pitch override precedence
- `tests/test_xlstm_counts_inference.py`
  - strike/ball count rules from `result`
  - walk/strikeout terminal states handled
- `tests/test_xlstm_decode.py`
  - tokens -> `Pitch` mapping and back
- `tests/test_xlstm_generation_mask.py`
  - grammar mask enforces valid next tokens

Update:

- `tests/test_api_predict_pitcher.py` to include `pa_id` in `prev_pitches`.

## Docs Updates

Update:

- `docs/rest-api.md`
- `docs/python-api.md`
- `docs/algorithms.md`

Add:

- `docs/xlstm_refactor_spec.md` (this document).

## Appendix: Count Inference Rules (Outline)

Rules apply in order, based on `Pitch.result`:

- Ball outcomes: `ball`, `ball_in_dirt`, `blocked_ball`, `automatic_ball`,
  `intentional_ball`, `pitchout`
  - increment balls unless already at 3
- Strike outcomes: `called_strike`, `swinging_strike`, `swinging_strike_blocked`,
  `swinging_pitchout`, `foul_tip`, `automatic_strike`
  - increment strikes unless already at 2
- Fouls: `foul`, `foul_bunt`, `bunt_foul_tip`, `foul_pitchout`
  - increment strikes only if strikes < 2
- In-play / hit-by-pitch:
  - `hit_into_play`, `hit_by_pitch`, `missed_bunt` end the PA
  - counts are reset on next `PA_START`

If a `result` is unknown or ambiguous, require explicit `count_balls/count_strikes`.
