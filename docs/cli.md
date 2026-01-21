# CLI Reference

PitchPredict provides a command-line interface for running the prediction server.

## Usage

```bash
pitchpredict <command> [options]
```

## Commands

### serve

Start the PitchPredict REST API server.

```bash
pitchpredict serve [options]
```

#### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--host` | `-H` | `0.0.0.0` | Host address to bind the server to |
| `--port` | `-p` | `8056` | Port number to bind the server to |
| `--reload` | `-r` | `false` | Enable auto-reload on code changes |

#### Examples

**Start with default settings:**

```bash
pitchpredict serve
```

Server starts at `http://0.0.0.0:8056`

**Specify host and port:**

```bash
pitchpredict serve --host 127.0.0.1 --port 8080
```

**Development mode with auto-reload:**

```bash
pitchpredict serve --reload
```

Or using short options:

```bash
pitchpredict serve -H 127.0.0.1 -p 8080 -r
```

## Server Output

When the server starts, you'll see output like:

```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8056 (Press CTRL+C to quit)
```

## Stopping the Server

Press `Ctrl+C` to stop the server:

```
^CINFO:     Shutting down
INFO:     Waiting for application shutdown.
INFO:     Application shutdown complete.
INFO:     Finished server process [12345]
```

## Help

View help information:

```bash
pitchpredict --help
pitchpredict serve --help
```
