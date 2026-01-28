# Data Columns Reference

This document lists all Statcast columns used by each similarity prediction mode.

## Pitcher Prediction Columns

The `PITCHER_PITCH_COLUMNS` tuple defines 57 columns fetched for pitcher predictions:

### Game Context

| Column | Type | Description |
|--------|------|-------------|
| `game_date` | string | Game date (YYYY-MM-DD) |
| `game_date_dt` | datetime | Parsed game date (computed) |
| `inning` | int | Inning number |
| `outs_when_up` | int | Outs when batter came up (0-2) |

### Count & Situation

| Column | Type | Description |
|--------|------|-------------|
| `balls` | int | Ball count (0-3) |
| `strikes` | int | Strike count (0-2) |
| `pitch_number` | int | Pitch number in plate appearance |
| `n_thruorder_pitcher` | int | Times through the order for pitcher |

### Players

| Column | Type | Description |
|--------|------|-------------|
| `batter` | int | Batter MLBAM ID |
| `p_throws` | L/R | Pitcher throwing hand |
| `stand` | L/R | Batter batting side |
| `age_pit` | int | Pitcher age |
| `age_bat` | int | Batter age |
| `batter_days_since_prev_game` | int | Batter's rest days |
| `pitcher_days_since_prev_game` | int | Pitcher's rest days |

### Score & Runners

| Column | Type | Description |
|--------|------|-------------|
| `bat_score` | int | Batting team score |
| `fld_score` | int | Fielding team score |
| `on_1b` | int/null | Runner on 1st (MLBAM ID or null) |
| `on_2b` | int/null | Runner on 2nd (MLBAM ID or null) |
| `on_3b` | int/null | Runner on 3rd (MLBAM ID or null) |

### Fielders

| Column | Type | Description |
|--------|------|-------------|
| `fielder_2` | int | Catcher MLBAM ID |
| `fielder_3` | int | First baseman MLBAM ID |
| `fielder_4` | int | Second baseman MLBAM ID |
| `fielder_5` | int | Third baseman MLBAM ID |
| `fielder_6` | int | Shortstop MLBAM ID |
| `fielder_7` | int | Left fielder MLBAM ID |
| `fielder_8` | int | Center fielder MLBAM ID |
| `fielder_9` | int | Right fielder MLBAM ID |

### Strike Zone

| Column | Type | Description |
|--------|------|-------------|
| `sz_top` | float | Top of strike zone (feet) |
| `sz_bot` | float | Bottom of strike zone (feet) |

### Pitch Characteristics

| Column | Type | Description |
|--------|------|-------------|
| `pitch_type` | string | Pitch type code (see below) |
| `release_speed` | float | Pitch velocity (mph) |
| `release_spin_rate` | float | Spin rate (rpm) |
| `spin_axis` | float | Spin axis (degrees) |
| `plate_x` | float | Horizontal plate location (feet) |
| `plate_z` | float | Vertical plate location (feet) |

### Release Point

| Column | Type | Description |
|--------|------|-------------|
| `release_pos_x` | float | Horizontal release position (feet) |
| `release_pos_z` | float | Vertical release position (feet) |
| `release_extension` | float | Release extension (feet) |

### Pitch Movement

| Column | Type | Description |
|--------|------|-------------|
| `vx0` | float | Velocity X component at 50ft |
| `vy0` | float | Velocity Y component at 50ft |
| `vz0` | float | Velocity Z component at 50ft |
| `ax` | float | Acceleration X component |
| `ay` | float | Acceleration Y component |
| `az` | float | Acceleration Z component |

### Outcome

| Column | Type | Description |
|--------|------|-------------|
| `type` | S/B/X | Pitch result (strike/ball/in-play) |
| `description` | string | Detailed pitch description |
| `events` | string | Plate appearance outcome (if any) |

### Batted Ball (when contact)

| Column | Type | Description |
|--------|------|-------------|
| `bb_type` | string | Batted ball type |
| `launch_speed` | float | Exit velocity (mph) |
| `launch_angle` | float | Launch angle (degrees) |
| `bat_speed` | float | Bat speed (mph) |
| `swing_length` | float | Swing length (feet) |

### Expected Stats

| Column | Type | Description |
|--------|------|-------------|
| `estimated_ba_using_speedangle` | float | xBA |
| `estimated_slg_using_speedangle` | float | xSLG |
| `estimated_woba_using_speedangle` | float | xwOBA |

---

## Batter Prediction Columns

The `BATTER_PITCH_COLUMNS` tuple defines 21 columns:

| Column | Type | Description |
|--------|------|-------------|
| `game_date` | string | Game date |
| `game_date_dt` | datetime | Parsed game date |
| `pitcher` | int | Pitcher MLBAM ID |
| `balls` | int | Ball count |
| `strikes` | int | Strike count |
| `bat_score` | int | Batting team score |
| `fld_score` | int | Fielding team score |
| `pitch_type` | string | Pitch type code |
| `release_speed` | float | Pitch velocity |
| `plate_x` | float | Horizontal location |
| `plate_z` | float | Vertical location |
| `type` | S/B/X | Pitch result |
| `bat_speed` | float | Bat speed |
| `swing_length` | float | Swing length |
| `events` | string | PA outcome |
| `bb_type` | string | Batted ball type |
| `launch_speed` | float | Exit velocity |
| `launch_angle` | float | Launch angle |
| `estimated_ba_using_speedangle` | float | xBA |
| `estimated_slg_using_speedangle` | float | xSLG |
| `estimated_woba_using_speedangle` | float | xwOBA |

---

## Batted Ball Prediction Columns

The `BATTED_BALL_COLUMNS` tuple defines 16 columns:

| Column | Type | Description |
|--------|------|-------------|
| `game_date` | string | Game date |
| `game_date_dt` | datetime | Parsed game date |
| `launch_speed` | float | Exit velocity (mph) |
| `launch_angle` | float | Launch angle (degrees) |
| `hc_x` | float | Hit coordinate X (spray) |
| `hc_y` | float | Hit coordinate Y (depth) |
| `bb_type` | string | Batted ball type |
| `outs_when_up` | int | Outs |
| `on_1b` | int/null | Runner on 1st |
| `on_2b` | int/null | Runner on 2nd |
| `on_3b` | int/null | Runner on 3rd |
| `batter` | int | Batter MLBAM ID |
| `events` | string | Outcome event |
| `estimated_ba_using_speedangle` | float | xBA |
| `estimated_slg_using_speedangle` | float | xSLG |
| `estimated_woba_using_speedangle` | float | xwOBA |

---

## Pitch Type Codes

Standard Statcast pitch type abbreviations:

| Code | Pitch Type |
|------|------------|
| `FF` | Four-seam fastball |
| `SI` | Sinker |
| `FC` | Cutter |
| `SL` | Slider |
| `CU` | Curveball |
| `CH` | Changeup |
| `FS` | Splitter |
| `KC` | Knuckle curve |
| `SV` | Sweeper |
| `ST` | Sweeping curve |
| `KN` | Knuckleball |
| `EP` | Eephus |
| `SC` | Screwball |
| `CS` | Slow curve |
| `FO` | Forkball |
| `FA` | Fastball (generic) |
| `PO` | Pitch out |
| `IN` | Intentional ball |
| `AB` | Automatic ball |
| `AS` | Automatic strike |
| `UN` | Unknown |

### Fastball vs Offspeed Classification

The algorithm groups pitches for detailed statistics:

**Fastballs:** `FF`, `FC`, `SI`

**Offspeed:** Everything else

---

## Batted Ball Types

| Code | Batted Ball Type |
|------|------------------|
| `ground_ball` | Ground ball |
| `line_drive` | Line drive |
| `fly_ball` | Fly ball |
| `popup` | Pop up |

---

## Outcome Event Types

Common events in the `events` column:

### Hits
- `single`
- `double`
- `triple`
- `home_run`

### Outs
- `field_out`
- `strikeout`
- `strikeout_double_play`
- `grounded_into_double_play`
- `double_play`
- `triple_play`
- `force_out`
- `fielders_choice`
- `fielders_choice_out`
- `sac_fly`
- `sac_fly_double_play`
- `sac_bunt`
- `sac_bunt_double_play`

### Other
- `field_error`
- `walk`
- `hit_by_pitch`
- `catcher_interf`
- `caught_stealing_2b`
- `caught_stealing_3b`
- `caught_stealing_home`
- `pickoff_1b`
- `pickoff_2b`
- `pickoff_3b`

---

## Type Column Values

The `type` column indicates pitch result:

| Value | Meaning |
|-------|---------|
| `S` | Strike (called, swinging, foul) |
| `B` | Ball |
| `X` | In play (contact made) |
