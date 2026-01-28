# CLI Reference

PitchPredict provides a command-line interface for running predictions, looking up players, managing the local cache, and starting the REST API server. Most commands call the local API directly and do not require the server to be running.

## Usage

```bash
pitchpredict <command> [options]
```

You can also run the module directly:

```bash
python -m pitchpredict.cli <command> [options]
```

## Top-level commands

| Command | Description |
|---------|-------------|
| `serve` | Start the PitchPredict REST API server |
| `predict` | Run pitcher, batter, or batted-ball predictions |
| `player` | Lookup players and fetch player details |
| `cache` | Inspect, clear, or warm the local cache |
| `version` | Show version information |

---

## serve

Start the PitchPredict REST API server.

```bash
pitchpredict serve [options]
```

### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--host` | `-H` | `0.0.0.0` | Host address to bind the server to |
| `--port` | `-p` | `8056` | Port number to bind the server to |
| `--reload` | `-r` | `false` | Enable auto-reload on code changes |

### Examples

```bash
pitchpredict serve
pitchpredict serve --host 127.0.0.1 --port 8080
pitchpredict serve -H 127.0.0.1 -p 8080 -r
```

---

## version

Show the installed PitchPredict version.

```bash
pitchpredict version
```

---

## predict

Run predictions for pitchers, batters, and batted balls.

```bash
pitchpredict predict <subcommand> [options]
```

### Common output options

Many prediction commands share these flags:

| Option | Default | Description |
|--------|---------|-------------|
| `--format` | `rich` | Output format (`rich` for tables, `json` for machine-readable) |
| `--verbose` | `false` | Show detailed output when available |
| `--algorithm` | `similarity` | Algorithm to use (`similarity` by default) |

> Note: The CLI defaults to the similarity algorithm. If you choose `--algorithm xlstm`, the CLI uses a cold-start request (`prev_pitches=[]`).

### predict pitcher

Predict the next pitch type, location, and outcome for a pitcher/batter matchup.

```bash
pitchpredict predict pitcher <pitcher> <batter> [options]
```

**Arguments**

- `pitcher`: Pitcher name or MLBAM ID
- `batter`: Batter name or MLBAM ID

**Options**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--balls` | `-b` | `0` | Ball count (0-3) |
| `--strikes` | `-s` | `0` | Strike count (0-2) |
| `--outs` | `-o` | — | Number of outs (0-2) |
| `--inning` | `-i` | — | Inning number |
| `--date` | `-d` | today | Game date (YYYY-MM-DD) |
| `--algorithm` | `-a` | `similarity` | Algorithm to use |
| `--format` | `-f` | `rich` | Output format (`rich` or `json`) |
| `--verbose` | `-v` | `false` | Show detailed output |

**Examples**

```bash
pitchpredict predict pitcher "Shohei Ohtani" "Aaron Judge"
pitchpredict predict pitcher 660271 592450 --date 2024-07-04 --balls 1 --strikes 2
pitchpredict predict pitcher "Zack Wheeler" "Juan Soto" --format json
```

### predict batter

Predict the outcome for a batter given an incoming pitch.

```bash
pitchpredict predict batter <batter> <pitcher> <type> <speed> <release_x> <release_z> [options]
```

**Arguments**

- `batter`: Batter name or MLBAM ID
- `pitcher`: Pitcher name or MLBAM ID
- `type`: Pitch type (e.g., `FF`, `SL`, `CU`)
- `speed`: Pitch speed in mph
- `release_x`: Pitch horizontal location
- `release_z`: Pitch vertical location

**Options**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--balls` | `-b` | `0` | Ball count (0-3) |
| `--strikes` | `-s` | `0` | Strike count (0-2) |
| `--score-bat` | — | `0` | Batting team score |
| `--score-fld` | — | `0` | Fielding team score |
| `--date` | `-d` | today | Game date (YYYY-MM-DD) |
| `--algorithm` | `-a` | `similarity` | Algorithm to use |
| `--format` | `-f` | `rich` | Output format (`rich` or `json`) |
| `--verbose` | `-v` | `false` | Show detailed output |

**Examples**

```bash
pitchpredict predict batter "Aaron Judge" "Gerrit Cole" FF 96.5 0.15 2.85
pitchpredict predict batter "Mookie Betts" "Corbin Burnes" SL 88.2 -0.2 2.4 --format json
```

### predict batted-ball

Predict the outcome of a batted ball given launch parameters.

```bash
pitchpredict predict batted-ball <launch_speed> <launch_angle> [options]
```

**Arguments**

- `launch_speed`: Exit velocity in mph
- `launch_angle`: Launch angle in degrees

**Options**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--spray-angle` | — | — | Spray angle in degrees |
| `--bb-type` | — | — | Batted ball type (`ground_ball`, `line_drive`, `fly_ball`, `popup`) |
| `--outs` | `-o` | — | Number of outs (0-2) |
| `--bases-state` | — | — | Base state (0-7, binary encoding of occupied bases) |
| `--batter-id` | — | — | Batter MLBAM ID |
| `--date` | `-d` | — | Game date (YYYY-MM-DD) |
| `--algorithm` | `-a` | `similarity` | Algorithm to use |
| `--format` | `-f` | `rich` | Output format (`rich` or `json`) |
| `--verbose` | `-v` | `false` | Show detailed output |

**Examples**

```bash
pitchpredict predict batted-ball 102.3 24
pitchpredict predict batted-ball 95.0 12 --spray-angle 10 --outs 1 --bases-state 3
pitchpredict predict batted-ball 88.4 -5 --format json
```

---

## player

Lookup player IDs and retrieve player metadata.

```bash
pitchpredict player <subcommand> [options]
```

### player lookup

Search for players by name.

```bash
pitchpredict player lookup <name> [options]
```

**Options**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--exact` | `-e` | `false` | Disable fuzzy matching |
| `--limit` | `-n` | `5` | Maximum number of results |
| `--format` | `-f` | `rich` | Output format (`rich` or `json`) |

**Examples**

```bash
pitchpredict player lookup "Juan Soto"
pitchpredict player lookup "Francisco Lindor" --exact --limit 3
pitchpredict player lookup "Devers" --format json
```

### player info

Get player metadata by MLBAM ID.

```bash
pitchpredict player info <mlbam_id> [options]
```

**Options**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--format` | `-f` | `rich` | Output format (`rich` or `json`) |

**Example**

```bash
pitchpredict player info 592450
```

---

## cache

Inspect and manage the local cache in `.pitchpredict_cache` (default).

```bash
pitchpredict cache <subcommand> [options]
```

### cache status

Show cache location, size, and contents summary.

```bash
pitchpredict cache status [options]
```

**Options**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--cache-dir` | — | `.pitchpredict_cache` | Cache directory |
| `--format` | `-f` | `rich` | Output format (`rich` or `json`) |

### cache clear

Clear cached data.

```bash
pitchpredict cache clear [options]
```

**Options**

| Option | Default | Description |
|--------|---------|-------------|
| `--cache-dir` | `.pitchpredict_cache` | Cache directory |
| `--confirm` | `false` | Skip confirmation prompt |
| `--category` | `all` | Category to clear (`pitcher`, `batter`, `batted_ball`, `players`, `all`) |

### cache warm

Pre-populate the cache for a player or for league-wide batted-ball data.

```bash
pitchpredict cache warm [player] [options]
```

**Options**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--type` | `-t` | `both` | Data to cache (`pitcher`, `batter`, `both`, `batted-ball`) |
| `--seasons` | `-n` | `3` | Seasons to fetch for batted-ball warming |
| `--cache-dir` | — | `.pitchpredict_cache` | Cache directory |

**Examples**

```bash
pitchpredict cache status
pitchpredict cache clear --category pitcher
pitchpredict cache clear --confirm
pitchpredict cache warm "Aaron Judge"
pitchpredict cache warm 592450 --type batter
pitchpredict cache warm --type batted-ball --seasons 4
```

---

## Help

```bash
pitchpredict --help
pitchpredict predict --help
pitchpredict predict pitcher --help
```
