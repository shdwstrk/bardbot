# Bardbot

Bardbot is a Discord voice bot for recording tabletop role-playing sessions. It saves separate WAV files for each speaker, transcribes them with `faster-whisper`, and can create campaign recaps using an Ollama model.

## Acknowledgment

The workflow of [Aitgray/BardBot](https://github.com/Aitgray/BardBot) inspired this project. The code here was independently generated and written; thank you to Aitgray for inspiring the project.

## License

This project's original code is licensed under the [MIT License](LICENSE). Third-party dependencies and other included materials remain subject to their respective licenses.

## What you need

- Docker.
- An NVIDIA GPU with compatible host drivers and NVIDIA Container Toolkit support. Transcription runs with CUDA and Whisper `large-v3`.
- A Discord bot token. Enable **Message Content Intent** in the [Discord Developer Portal](https://discord.com/developers/applications) and invite the bot with permission to read and send messages, connect to and speak in voice channels, and attach files.
- An Ollama server reachable from the container if you want to use `!summarize`. Recording and transcription do not require Ollama.

Python, CUDA runtime libraries, `ffmpeg`, and the bot's Python packages are installed in the Docker image; you do not need to install them on the host.

## Build and run

Build the image from the project directory:

```bash
docker build -t bardbot .
```

Create host directories for persistent recordings and transcripts, then run the container:

```bash
mkdir -p recordings transcripts
docker run --rm --gpus all \
  -e DISCORD_TOKEN="your-discord-bot-token" \
  -v "$PWD/recordings:/app/recordings" \
  -v "$PWD/transcripts:/app/transcripts" \
  bardbot
```

The bind mounts preserve recordings and transcripts after the container stops. Keep the bot token private; do not put it in source control or the image.

The image uses NVIDIA CUDA. `--gpus all` requires a working NVIDIA Container Toolkit installation on the Docker host.
Bot output is written to stdout and stderr, which Docker captures as container logs. View it with `docker logs -f <container>` (find the container name or ID with `docker ps`). Python output is unbuffered in the image, so messages appear in the logs as they are written.

### Docker Compose

For a reusable setup, create a `.env` file in the project directory:

```dotenv
DISCORD_TOKEN=your-discord-bot-token
```

Then build and start the bot:

```bash
docker compose up --build -d
```

Compose starts Bardbot and Ollama on a dedicated `bardbot-network` bridge network. Bardbot waits for the Ollama server, then asks it to pull the configured summary model (default `llama3.1:latest`) before connecting to Discord. The model is stored in Ollama's named volume. Follow startup with `docker compose logs -f bardbot` to see when the pull starts and completes. Recordings and transcripts are mounted from the project directory. Bardbot and Ollama request access to all NVIDIA GPUs; the host must have Docker Compose and a working NVIDIA Container Toolkit installation.

To use a different model, set `OLLAMA_MODEL` in `.env`. For example:

```dotenv
OLLAMA_MODEL=qwen2.5:latest
```

Stop the services with `docker compose down` to remove their containers and network. The `.env` file is ignored by Git; keep your token private.

### Optional character names

To use character or player names in summaries, copy `character_names_template.json` to `character_names.json`, replace the example IDs with Discord user IDs, and mount the file read-only:

```bash
docker run --rm --gpus all \
  -e DISCORD_TOKEN="your-discord-bot-token" \
  -v "$PWD/recordings:/app/recordings" \
  -v "$PWD/transcripts:/app/transcripts" \
  -v "$PWD/character_names.json:/app/character_names.json:ro" \
  bardbot
```

The JSON file maps Discord user IDs (as strings) to display names. Without it, summaries use user IDs.

### Ollama

The default summary model is `llama3.1:latest`. With Compose, Bardbot asks Ollama to pull the configured model at startup. For an Ollama server running outside Compose, you can enable the same behavior by setting `OLLAMA_AUTO_PULL=true`. Otherwise, make sure the model is available there:

```bash
ollama pull llama3.1:latest
```

The bot connects to the Compose Ollama service by default. For an Ollama server running outside Compose, set `OLLAMA_HOST` to that server's hostname or IP address. With `docker run` on Linux, for example, add a host-gateway mapping and set the host:

```bash
docker run --rm --gpus all \
  --add-host=host.docker.internal:host-gateway \
  -e DISCORD_TOKEN="your-discord-bot-token" \
  -e OLLAMA_HOST=host.docker.internal \
  -v "$PWD/recordings:/app/recordings" \
  -v "$PWD/transcripts:/app/transcripts" \
  bardbot
```

## Discord commands

Commands use the `!` prefix.

| Command | Description |
| --- | --- |
| `!ping` | Check that the bot responds. |
| `!join` | Join your current voice channel, or move to it if already connected. |
| `!start_recording` | Start recording each speaker to a separate WAV file. |
| `!stop_recording` | Stop the active recording and save its files. |
| `!leave` | Stop an active recording, if any, and disconnect. |
| `!list_sessions` | List recorded session directories. |
| `!transcribe [session]` | Transcribe a recorded session. |
| `!summarize [session]` | Summarize an existing transcript and send the summary file. |
| `!recap` | Create a beat-by-beat recap of the latest session's transcript and send the recap file. |

For `!transcribe` and `!summarize`, provide a session directory name to select a session. If omitted, the bot selects an entry from the recordings directory. Transcribe a session before summarizing it.

## Configuration

Set these environment variables with Docker's `-e NAME=value` option:

| Variable | Default | Purpose |
| --- | --- | --- |
| `DISCORD_TOKEN` | Empty | Discord bot token; required. |
| `RECORDINGS_DIR` | `recordings` | Directory for timestamped recording sessions. |
| `TRANSCRIPTS_DIR` | `transcripts` | Directory for transcript and summary files. |
| `OLLAMA_HOST` | `localhost` | Ollama hostname or IP address, without a URL scheme. |
| `OLLAMA_PORT` | `11434` | Ollama server port. |
| `OLLAMA_TIMEOUT_SECONDS` | `360` | Timeout for a summary request. |
| `OLLAMA_MODEL` | `llama3.1:latest` | Ollama model used for summaries. |
| `OLLAMA_AUTO_PULL` | `false` | Ask Ollama to pull the configured model before the bot connects to Discord. |
| `OLLAMA_STARTUP_TIMEOUT_SECONDS` | `120` | How long to wait for Ollama to become available when auto-pull is enabled. |
| `OLLAMA_PULL_TIMEOUT_SECONDS` | `3600` | Maximum time to wait for the model pull to complete. |

Transcripts are newline-delimited JSON records with speaker ID, start and end times, and text. Summaries are saved as `<session>_summary.txt`.

## DAVE voice encryption

This project currently pins Pycord to commit [`5088a0f`](https://github.com/Pycord-Development/pycord/commit/5088a0fb7419180e890e2627568a16131064483b), an unmerged experimental patch for DAVE voice reception. It is not an official Pycord release and may need to be replaced as upstream support evolves. Rebuild the Docker image after changing this dependency.

Test recording in a DAVE-enabled channel before relying on it. The bot saves unencrypted WAV audio after receiving and decoding it; obtain participants' consent before recording.

## Future development

Potential areas for future development include:

- Add options for incorporation of user provided notes to pass to Ollama.
- Replace the experimental Pycord pin with an official release once it supports DAVE voice reception.
- Make transcription settings, such as the Whisper model, language, and compute device, configurable.
- Improve speaker and character attribution beyond the current Discord-user name mapping.
- Add automated tests for recording, transcription, and summary workflows.
- Refine LLM prompts
- Add functionality to auto configure AI models based on available resources.
- Add functionality to use older session data when creating summaries and recaps.

## Privacy

Recording captures voice-channel participants. Obtain their consent before recording. Audio and transcript files are stored in the mounted directories; when `!summarize` is used, transcript text is sent to the configured Ollama server. All data is processed locally, unless an external Ollama server is used. Some calls to Hugging Face to update the Whisper model will occur on first run.
