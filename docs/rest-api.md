# REST API Reference

PitchPredict includes a FastAPI server for making predictions via HTTP requests.

## Starting the Server

### Via CLI

```bash
pitchpredict serve
```

Options:

| Option | Default | Description |
|--------|---------|-------------|
| `--host`, `-H` | `0.0.0.0` | Host to bind to |
| `--port`, `-p` | `8056` | Port to bind to |
| `--reload`, `-r` | `false` | Enable hot reload for development |

Example:

```bash
pitchpredict serve --host 127.0.0.1 --port 8080 --reload
```

### Via Python

```python
from pitchpredict.server import run_server

run_server(host="0.0.0.0", port=8056, reload=False)
```

---

## Endpoints

### GET /

Health check and server information.

**Response:**

```json
{
  "name": "pitchpredict",
  "version": "0.3.0",
  "status": "running",
  "uptime": "0:05:32.123456"
}
```

**Example:**

```bash
curl http://localhost:8056/
```

---

### POST /predict/pitcher

Predict the pitcher's next pitch.

**Request Body:**

```json
{
  "pitcher_name": "Clayton Kershaw",
  "batter_name": "Aaron Judge",
  "balls": 0,
  "strikes": 0,
  "score_bat": 0,
  "score_fld": 0,
  "game_date": "2024-06-15",
  "algorithm": "similarity"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `pitcher_name` | string | Yes | Pitcher's name |
| `batter_name` | string | Yes | Batter's name |
| `balls` | integer | Yes | Ball count (0-3) |
| `strikes` | integer | Yes | Strike count (0-2) |
| `score_bat` | integer | Yes | Batting team's score |
| `score_fld` | integer | Yes | Fielding team's score |
| `game_date` | string | Yes | Date in "YYYY-MM-DD" format |
| `algorithm` | string | Yes | `"similarity"` or `"deep"` |

**Response:**

```json
{
  "basic_pitch_data": {
    "pitch_type_probs": {
      "FF": 0.45,
      "SL": 0.30,
      "CU": 0.15,
      "CH": 0.10
    },
    "pitch_speed_mean": 92.5,
    "pitch_speed_std": 4.2,
    "pitch_x_mean": 0.12,
    "pitch_x_std": 0.85,
    "pitch_z_mean": 2.45,
    "pitch_z_std": 0.72
  },
  "detailed_pitch_data": {
    "pitch_prob_fastball": 0.55,
    "pitch_prob_offspeed": 0.45,
    "pitch_data_fastballs": { ... },
    "pitch_data_offspeed": { ... },
    "pitch_data_overall": { ... }
  },
  "basic_outcome_data": {
    "outcome_probs": {
      "strike": 0.45,
      "ball": 0.35,
      "contact": 0.20
    },
    "swing_probability": 0.55,
    "swing_event_probs": { ... },
    "contact_probability": 0.20,
    "contact_event_probs": { ... }
  },
  "detailed_outcome_data": {
    "swing_data": { ... },
    "contact_data": { ... }
  },
  "prediction_metadata": {
    "start_time": "2024-06-15T10:30:00",
    "end_time": "2024-06-15T10:30:02",
    "duration": 2.5,
    "n_pitches_total": 5000,
    "n_pitches_sampled": 250,
    "sample_pctg": 0.05
  }
}
```

**Example:**

```bash
curl -X POST http://localhost:8056/predict/pitcher \
  -H "Content-Type: application/json" \
  -d '{
    "pitcher_name": "Clayton Kershaw",
    "batter_name": "Aaron Judge",
    "balls": 1,
    "strikes": 2,
    "score_bat": 0,
    "score_fld": 1,
    "game_date": "2024-06-15",
    "algorithm": "similarity"
  }'
```

---

### POST /predict/batter

Predict the batter's outcome for a given pitch.

**Request Body:**

```json
{
  "batter_name": "Aaron Judge",
  "pitcher_name": "Clayton Kershaw",
  "balls": 1,
  "strikes": 2,
  "score_bat": 0,
  "score_fld": 1,
  "game_date": "2024-06-15",
  "pitch_type": "FF",
  "pitch_speed": 95.0,
  "pitch_x": 0.5,
  "pitch_y": 2.5,
  "algorithm": "similarity"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `batter_name` | string | Yes | Batter's name |
| `pitcher_name` | string | Yes | Pitcher's name |
| `balls` | integer | Yes | Ball count (0-3) |
| `strikes` | integer | Yes | Strike count (0-2) |
| `score_bat` | integer | Yes | Batting team's score |
| `score_fld` | integer | Yes | Fielding team's score |
| `game_date` | string | Yes | Date in "YYYY-MM-DD" format |
| `pitch_type` | string | Yes | Pitch type code (e.g., "FF") |
| `pitch_speed` | float | Yes | Pitch speed in mph |
| `pitch_x` | float | Yes | Horizontal location (feet from center) |
| `pitch_y` | float | Yes | Vertical location (feet from ground) |
| `algorithm` | string | Yes | `"similarity"` or `"deep"` |

**Response:**

```json
{
  "basic_outcome_data": {
    "outcome_probs": {
      "strike": 0.40,
      "ball": 0.30,
      "contact": 0.30
    },
    "swing_probability": 0.65,
    "swing_event_probs": { ... },
    "contact_probability": 0.30,
    "contact_event_probs": { ... }
  },
  "detailed_outcome_data": {
    "swing_data": { ... },
    "contact_data": { ... }
  },
  "prediction_metadata": {
    "start_time": "2024-06-15T10:30:00",
    "end_time": "2024-06-15T10:30:01",
    "duration": 1.5,
    "n_pitches_total": 3000,
    "n_pitches_sampled": 150,
    "sample_pctg": 0.05
  }
}
```

**Example:**

```bash
curl -X POST http://localhost:8056/predict/batter \
  -H "Content-Type: application/json" \
  -d '{
    "batter_name": "Aaron Judge",
    "pitcher_name": "Clayton Kershaw",
    "balls": 1,
    "strikes": 2,
    "score_bat": 0,
    "score_fld": 1,
    "game_date": "2024-06-15",
    "pitch_type": "FF",
    "pitch_speed": 95.0,
    "pitch_x": 0.5,
    "pitch_y": 2.5,
    "algorithm": "similarity"
  }'
```

---

### POST /predict/batted-ball

Predict batted ball outcome probabilities given exit velocity, launch angle, and optional game context.

**Request Body:**

```json
{
  "launch_speed": 95.0,
  "launch_angle": 18.0,
  "algorithm": "similarity",
  "spray_angle": 10.0,
  "bb_type": "line_drive",
  "outs": 1,
  "bases_state": 5,
  "batter_id": 592450,
  "game_date": "2024-06-15"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `launch_speed` | float | Yes | Exit velocity in mph |
| `launch_angle` | float | Yes | Launch angle in degrees (-90 to 90) |
| `algorithm` | string | Yes | `"similarity"` (only option for now) |
| `spray_angle` | float | No | Horizontal direction (-45 to 45, 0 = center field) |
| `bb_type` | string | No | Batted ball type: `"ground_ball"`, `"line_drive"`, `"fly_ball"`, `"popup"` |
| `outs` | integer | No | Current outs (0-2) |
| `bases_state` | integer | No | Bases occupied bitmask (1=1B, 2=2B, 4=3B, e.g., 5 = runners on 1B and 3B) |
| `batter_id` | integer | No | MLBAM batter ID |
| `game_date` | string | No | Date in "YYYY-MM-DD" format |

**Response:**

```json
{
  "basic_outcome_data": {
    "outcome_probs": {
      "single": 0.18,
      "double": 0.08,
      "triple": 0.01,
      "home_run": 0.05,
      "groundout": 0.25,
      "flyout": 0.20,
      "lineout": 0.08,
      "popout": 0.02,
      "sac_fly": 0.03,
      "double_play": 0.04,
      "field_error": 0.02,
      "force_out": 0.04
    },
    "hit_probability": 0.32,
    "xba": 0.285,
    "bb_type_inferred": "line_drive"
  },
  "detailed_outcome_data": {
    "sample_launch_speed_mean": 98.5,
    "sample_launch_angle_mean": 15.2,
    "expected_stats": {
      "xBA": 0.285,
      "xSLG": 0.520,
      "xwOBA": 0.380
    }
  },
  "prediction_metadata": {
    "n_batted_balls_sampled": 750,
    "sample_pctg": 0.05,
    "similarity_weights": {
      "launch_speed": 0.45,
      "launch_angle": 0.40,
      "spray_angle": 0.05,
      "bases_state": 0.05,
      "outs": 0.03,
      "date": 0.02
    }
  }
}
```

**Context-Aware Outcome Filtering:**

When game context is provided, outcomes are filtered to only include logically possible results:

| Outcome | Condition |
|---------|-----------|
| `sac_fly` | `outs < 2` AND runner on 3B (`bases_state & 4`) |
| `double_play` | At least one runner on base (`bases_state > 0`) |
| `force_out` | Force play possible (runner on 1B) |

If no game context is provided, all outcomes are included with full historical probabilities.

**Example:**

```bash
curl -X POST http://localhost:8056/predict/batted-ball \
  -H "Content-Type: application/json" \
  -d '{
    "launch_speed": 95.0,
    "launch_angle": 18.0,
    "algorithm": "similarity"
  }'
```

**Example with game context:**

```bash
curl -X POST http://localhost:8056/predict/batted-ball \
  -H "Content-Type: application/json" \
  -d '{
    "launch_speed": 102.0,
    "launch_angle": 25.0,
    "algorithm": "similarity",
    "outs": 1,
    "bases_state": 5
  }'
```

---

## OpenAPI Documentation

When the server is running, interactive API documentation is available at:

- **Swagger UI:** `http://localhost:8056/docs`
- **ReDoc:** `http://localhost:8056/redoc`

---

## Error Responses

### 400 Bad Request

Invalid request parameters.

```json
{
  "detail": "unrecognized algorithm: invalid_algorithm"
}
```

### 500 Internal Server Error

Server-side error (e.g., player not found, data fetch failed).

```json
{
  "detail": "Player not found: Unknown Player"
}
```

---

## CORS

By default, the server does not configure CORS. To enable cross-origin requests for a web frontend, you can wrap the FastAPI app:

```python
from fastapi.middleware.cors import CORSMiddleware
from pitchpredict.server import app

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```
