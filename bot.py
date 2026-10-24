import faster_whisper as whisper
import tqdm
import discord
import os
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
import requests

from discord.ext import commands
from recording import DaveWaveSink
from transcript_chunks import TranscriptSegment, chunk_transcript, read_transcript

BASE_DIR = Path(__file__).resolve().parent
DISCORD_TOKEN = str(os.getenv("DISCORD_TOKEN", ""))
RECORDINGS_DIR = Path( BASE_DIR / os.getenv("RECORDINGS_DIR", "recordings"))
TEXT_FILES_DIR = Path(BASE_DIR / os.getenv("TEXT_FILES_DIR", "text"))
TRANSCRIPTS_DIR = TEXT_FILES_DIR / "transcripts"
SUMMARIES_DIR = TEXT_FILES_DIR / "summaries"
RECAPS_DIR = TEXT_FILES_DIR / "recaps"
PROMPTS_DIR = BASE_DIR / "prompts"
OLLAMA_HOST = str(os.getenv("OLLAMA_HOST", "localhost"))
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "360"))
OLLAMA_MODEL = str(os.getenv("OLLAMA_MODEL", "llama3.1:latest"))
OLLAMA_AUTO_PULL = os.getenv("OLLAMA_AUTO_PULL", "false").lower() in {"1", "true", "yes"}
OLLAMA_STARTUP_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_STARTUP_TIMEOUT_SECONDS", "120"))
OLLAMA_PULL_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_PULL_TIMEOUT_SECONDS", "3600"))

# Load optional Discord-user display names for transcript summaries.
if(Path("character_names.json").exists()):
    with open("character_names.json") as f:
        CHARACTER_MAP = json.load(f)
else:
    CHARACTER_MAP = {}

def _load_prompt(prompt_name: str) -> str:
    """Load system instructions for an Ollama generation request."""
    prompt = (PROMPTS_DIR / prompt_name).read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"Prompt file {prompt_name} must not be empty.")
    return prompt


class BardBotClient(commands.Cog):
    def __init__(self, bot):
        """Store the bot and initialize voice-recording state."""
        self.bot = bot
        self.voice_client = None
        self.recording_sink = None

    @commands.command()
    async def ping(self, ctx):
        """Check if the bot is responsive."""
        await ctx.send('Pong!')
        print('Responded to ping command.')
    
    @commands.command()
    async def join(self, ctx):
        """Join the voice channel of the user who invoked the command."""
        if ctx.author.voice is None:
            await ctx.send('You are not in a voice channel!')
            print('Failed to join voice channel: User not in a voice channel!')
        else:
            if self.voice_client is None:
                self.voice_client = await ctx.author.voice.channel.connect()
                print(f'Joined {ctx.author.voice.channel.name}')
            else:
                await self.voice_client.move_to(ctx.author.voice.channel)
                print(f'Moved to {ctx.author.voice.channel.name}')

    @commands.command()
    async def leave(self, ctx):
        """Leave the current voice channel."""
        if self.voice_client:
            if self.voice_client.is_recording():
                self.voice_client.stop_recording()
                print('Stopped recording before disconnecting.')
            await self.voice_client.disconnect()
            await ctx.send('Disconnected from the voice channel!')
            print('Disconnected from the voice channel!')
        else:
            await ctx.send('I am not in a voice channel!')
            print('Failed to disconnect: Not in a voice channel!')
    
    @commands.command()
    async def start_recording(self, ctx):
        """Record each speaker to a local WAV file."""
        if not self.voice_client or not self.voice_client.is_connected():
            await ctx.send('I am not in a voice channel!')
            print('Failed to start recording: Not in a voice channel!')
            return
        else:
            if self.voice_client.is_recording():
                await ctx.send('Already recording!')
                print('Failed to start recording: Already recording!')
                return
            sink = DaveWaveSink()
            self.recording_sink = sink
            try:
                self.voice_client.start_recording(sink, self._recording_finished)
            except Exception:
                self.recording_sink = None
                raise
            await ctx.send('Recording started!')
    
    def _recording_finished(self, error):
        """Persist each speaker's completed WAV buffer in a timestamped session directory."""
        sink = self.recording_sink
        if sink is None:
            raise RuntimeError('Recording finished without an active sink.')
        self.recording_sink = None

        if error is not None:
            print(f'Recording stopped with an error: {error!r}')

        sink.cleanup()
        output_dir = RECORDINGS_DIR / datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)
        print("number of users recorded:", len(sink.audio_data.items()))
        for user_id, audio in sink.audio_data.items():
            print("User ID:", user_id)
            filename = f"{user_id}.wav"
            file_path = output_dir / filename
            with open(file_path, "wb") as f:
                f.write(audio.file.getbuffer())
            print(f'Saved recording for {user_id} to {file_path}')
        print(f'Recording finished; files saved to {output_dir}')
    
            
    @commands.command()
    async def stop_recording(self, ctx):
        """Stop recording audio and save to files."""
        if not self.voice_client or not self.voice_client.is_connected():
            await ctx.send('I am not in a voice channel!')
            print('Failed to stop recording: Not in a voice channel!')
            return
        if not self.voice_client.is_recording():
            await ctx.send('No recording in progress.')
            return
        else:
            self.voice_client.stop_recording()
            await ctx.send('Recording stopped!')

    @staticmethod
    def _latest_recording_session() -> Path | None:
        """Return the newest timestamped recording directory, if one exists."""
        if not RECORDINGS_DIR.is_dir():
            return None
        sessions = (entry for entry in RECORDINGS_DIR.iterdir() if entry.is_dir())
        return max(sessions, key=lambda entry: entry.name, default=None)
    
    @commands.command()
    async def transcribe(self, ctx, session: str = ""):
        """Transcribe audio. User can provide a session ID to transcribe or bot will use the most recent session by default."""
        await ctx.send('Transcribing audio... This may take a while depending on the length of the recording.')
        if session == "":
            latest_session = self._latest_recording_session()
            if latest_session is None:
                await ctx.send('No recorded sessions found.')
                print('No recorded sessions found.')
                return
            session = latest_session.name
        if not Path(RECORDINGS_DIR / session).exists():
            await ctx.send(f'Session {session} does not exist.')
            print(f'Session {session} does not exist.')
            return
        session_dir = Path(RECORDINGS_DIR / session )
        print('Transcribing audio... This may take a while depending on the length of the recording.')
        await asyncio.to_thread(self._transcribe_session, session, session_dir)
        await ctx.send(f'Transcription completed for session {session}. Transcript saved.')

    @staticmethod
    def _transcribe_session(session: str, session_dir: Path):
        """Run Whisper and persist a session transcript outside the Discord event loop."""
        model = whisper.WhisperModel("large-v3", device="cuda", compute_type="float16")
        transcript = []

        # Transcribe each speaker's WAV independently, retaining the filename's user ID.
        for audio_file in session_dir.glob("*.wav"):
            print(f'Transcribing {audio_file}...')
            with open(audio_file, "rb") as f:
                segments, info = model.transcribe(
                    f,
                    language="en",
                    beam_size=5,
                    condition_on_previous_text=False,
                    no_speech_threshold=0.6,
                    vad_filter=True,
                    vad_parameters=dict(
                        min_speech_duration_ms=500,
                        max_speech_duration_s=20.0,
                        min_silence_duration_ms=1000
                    )
                )
                with tqdm.tqdm(total=round(info.duration, 2), unit="s",desc=f"Transcribing {audio_file.name}") as pbar:
                    for segment in segments:
                        text = segment.text.strip()
                        transcript.append({
                            "user_id": audio_file.stem,
                            "start": segment.start,
                            "end": segment.end,
                            "text": text
                        })
                        pbar.update(segment.end - pbar.n)
            print(f'Transcription completed for {audio_file}.')

        # Put all speakers on one timeline, then combine consecutive entries for the same speaker.
        print(f'Transcription completed for session {session}.')
        print(f"Sorting and merging transcript segments for session {session}...")
        transcript.sort(key=lambda x: (x["start"], x["end"]))
        merged_transcript = []
        for segment in transcript:
            if merged_transcript and merged_transcript[-1]["user_id"] == segment["user_id"]:
                previous = merged_transcript[-1]
                previous["text"] = f'{previous["text"].rstrip()} {segment["text"].lstrip()}'.strip()
                previous["start"] = min(previous["start"], segment["start"])
                previous["end"] = max(previous["end"], segment["end"])
            else:
                merged_transcript.append(segment)
        transcript_file = TRANSCRIPTS_DIR / f"{session}.txt"
        TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(transcript_file, "w") as f:
            for segment in merged_transcript:
                f.write(json.dumps(segment) + "\n")
        print(f'Saving transcript for session {session}...')
    
    async def _extract_transcript_facts(
        self,
        ctx: commands.Context,
        segments: Sequence[TranscriptSegment],
        extraction_prompt: str,
    ) -> str:
        """Extract source-grounded facts from overlapping transcript chunks."""
        chunks = chunk_transcript(segments)
        extracted_facts = []
        progress_message = await ctx.send(
            f"Extracting transcript facts (0/{len(chunks)} chunks)..."
        )
        for index, chunk in enumerate(chunks, start=1):
            chunk_text = "\n".join(segment.render() for segment in chunk)
            await progress_message.edit(
                content=(
                    f"Extracting transcript facts ({index}/{len(chunks)} chunks; "
                    f"{len(chunk_text):,} characters in this chunk)..."
                )
            )
            started_at = time.monotonic()
            print(
                f"Starting Ollama extraction for chunk {index}/{len(chunks)} "
                f"({len(chunk_text):,} characters)."
            )
            try:
                response = await self.ask_ollama(ctx, extraction_prompt, chunk_text)
            except (TimeoutError, requests.RequestException):
                await progress_message.edit(
                    content=(
                        f"Transcript extraction stopped at chunk "
                        f"{index}/{len(chunks)}. See the bot logs for details."
                    )
                )
                raise
            facts = (response.json().get("response") or "").strip()
            if not facts:
                raise ValueError(f"Ollama returned no facts for transcript chunk {index}.")
            extracted_facts.append(f"Chunk {index}:\n{facts}")
            elapsed = time.monotonic() - started_at
            print(
                f"Finished Ollama extraction for chunk {index}/{len(chunks)} "
                f"in {elapsed:.1f}s."
            )
            await progress_message.edit(
                content=(
                    f"Extracted transcript facts ({index}/{len(chunks)} chunks; "
                    f"last chunk took {elapsed:.1f}s)."
                )
            )
        return "\n\n".join(extracted_facts)

    @commands.command()
    async def summarize(self, ctx, session: str = ""):
        """Summarize transcribed audio. User can provide a session ID to summarize or bot will use the most recent session by default."""
        await ctx.send('Summarizing audio...')
        print('Summarizing audio...')
        if session == "":
            latest_session = self._latest_recording_session()
            if latest_session is None:
                await ctx.send('No recorded sessions found.')
                print('No recorded sessions found.')
                return
            session = latest_session.name
        transcript_file = Path(TRANSCRIPTS_DIR / f"{session}.txt")
        if not transcript_file.exists():
            await ctx.send(f'Transcript for session {session} does not exist. Please transcribe the session first.')
            print(f'Transcript for session {session} does not exist. Please transcribe the session first.')
            return
        segments = read_transcript(transcript_file, CHARACTER_MAP)
        if sum(len(segment.text) for segment in segments) < 10:
            await ctx.send(f'Transcript for session {session} is too short to summarize or is empty.')
            print(f'Transcript for session {session} is too short to summarize or is empty.')
            return
        await ctx.send(f'Summarizing transcript for session {session}...')
        print(f'Summarizing transcript for session {session}...')
        try:
            system_prompt = _load_prompt("summarize.txt")
            extraction_prompt = _load_prompt("extract.txt")
        except (OSError, ValueError) as error:
            print(f"Failed to load summarization prompt: {error}")
            await ctx.send("Unable to load a summarization prompt. Check prompts/summarize.txt and prompts/extract.txt.")
            return
        extracted_facts = await self._extract_transcript_facts(
            ctx, segments, extraction_prompt
        )
        await ctx.send("All transcript chunks extracted. Generating the summary...")
        response = await self.ask_ollama(ctx, system_prompt, extracted_facts)
        summary = (response.json().get("response") or "").strip()
        summary_file = SUMMARIES_DIR / f"{session}.md"
        SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
        with open(summary_file, "w") as f:
            f.write(summary)
            print(f'Summary for session {session} saved to {summary_file}.')
        
        await ctx.send(file=discord.File(summary_file))
        
    @commands.command()
    async def list_sessions(self, ctx):
        """List all recorded sessions IDs."""
        sessions = sorted(
            (entry.name for entry in RECORDINGS_DIR.iterdir() if entry.is_dir())
        ) if RECORDINGS_DIR.is_dir() else []
        if not sessions:
            await ctx.send('No recorded sessions found.')
            print('No recorded sessions found.')
            return
        session_list = "\n".join(sessions)
        await ctx.send(f'Available recorded sessions:\n{session_list}')
        print("Session list sent to channel.")

    @commands.command()
    async def recap(self, ctx):
        """Recalls the last session.
        Gives a more play-by-play style recap."""
        latest_session = self._latest_recording_session()
        if latest_session is None:
            await ctx.send('No recorded sessions found.')
            print('No recorded sessions found.')
            return
        session = latest_session.name
        transcript_file = Path(TRANSCRIPTS_DIR / f"{session}.txt")
        if not transcript_file.exists():
            await ctx.send(f'Transcript for session {session} does not exist. Please transcribe the session first.')
            print(f'Transcript for session {session} does not exist. Please transcribe the session first.')
            return
        segments = read_transcript(transcript_file, CHARACTER_MAP)
        if sum(len(segment.text) for segment in segments) < 10:
            await ctx.send(f'Transcript for session {session} is too short to summarize or is empty.')
            print(f'Transcript for session {session} is too short to summarize or is empty.')
            return
        await ctx.send(f'Recapping transcript for session {session}...')
        print(f'Recapping transcript for session {session}...')
        try:
            system_prompt = _load_prompt("recap.txt")
            extraction_prompt = _load_prompt("extract.txt")
        except (OSError, ValueError) as error:
            print(f"Failed to load recap prompt: {error}")
            await ctx.send("Unable to load a recap prompt. Check prompts/recap.txt and prompts/extract.txt.")
            return
        extracted_facts = await self._extract_transcript_facts(
            ctx, segments, extraction_prompt
        )
        await ctx.send("All transcript chunks extracted. Generating the recap...")
        response = await self.ask_ollama(ctx, system_prompt, extracted_facts)
        recap = (response.json().get("response") or "").strip()
        recap_file = RECAPS_DIR / f"{session}.md"
        RECAPS_DIR.mkdir(parents=True, exist_ok=True)
        with open(recap_file, "w") as f:
            f.write(recap)
            print(f'Recap for session {session} saved to {recap_file}.')
        await ctx.send(file=discord.File(recap_file))

    async def ask_ollama(
        self, ctx, system_prompt: str, transcript_data: str
    ) -> requests.Response:
        """Send a non-streaming generation request to the configured Ollama server."""
        payload = {
            "model": OLLAMA_MODEL,
            "system": system_prompt,
            "prompt": transcript_data,
            "stream": False,
            "options": {
                #"num_ctx": 65536,
                "temperature": 0.2,
                "kv_cache_type":"q8_0", #default cache type overflows my 3080 Ti, so this reduces the size to fit completely on the GPU.
                #"num_predict":2048
            }
        }
        try:
            response = await asyncio.to_thread(
                requests.post,
                f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate",
                json=payload,
                timeout=OLLAMA_TIMEOUT_SECONDS,
            )
        except requests.exceptions.ReadTimeout as error:
            print(f"Ollama timed out after {OLLAMA_TIMEOUT_SECONDS:.0f} seconds processing request.")
            await ctx.send(
                f"Ollama did not finish processing request within {OLLAMA_TIMEOUT_SECONDS:.0f} seconds. "
                "Try again, increase OLLAMA_TIMEOUT_SECONDS, or use a smaller model/context."
            )
            raise TimeoutError("Ollama request timed out.") from error
        response.raise_for_status()
        return response

    @commands.Cog.listener()
    async def on_command_error(self, ctx: commands.Context, error: commands.CommandError):
        """A global error handler for all prefixed commands."""
        
        # Prevent double-firing if a command has its own local error handler
        if hasattr(ctx.command, 'on_error'):
            return

        # Catch when a user triggers an unknown command
        if isinstance(error, commands.CommandNotFound):
            await ctx.send(f"`{ctx.invoked_with}` is not a valid command. Type `!help` for assistance.")
            return

        # Catch missing required arguments
        if isinstance(error, commands.MissingRequiredArgument):
            if ctx.command is not None:
                await ctx.send(f"Missing parameters. Usage: `{ctx.prefix}{ctx.command.qualified_name} {ctx.command.signature}`")
            else:
                await ctx.send("Missing parameters.")
            return

        # Fallback to printing unhandled exceptions to your Unraid/terminal log
        print(f"Unhandled exception in command '{ctx.command}':", error)

        
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

async def pull_ollama_model():
    """Wait for Ollama, then ask it to pull the configured model."""
    base_url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
    deadline = asyncio.get_running_loop().time() + OLLAMA_STARTUP_TIMEOUT_SECONDS

    while True:
        try:
            response = await asyncio.to_thread(
                requests.get,
                f"{base_url}/api/version",
                timeout=5,
            )
            response.raise_for_status()
            break
        except requests.exceptions.RequestException as error:
            if asyncio.get_running_loop().time() >= deadline:
                raise RuntimeError(
                    f"Ollama did not become available at {base_url} "
                    f"within {OLLAMA_STARTUP_TIMEOUT_SECONDS:.0f} seconds."
                ) from error
            await asyncio.sleep(2)

    print(f"Pulling Ollama model {OLLAMA_MODEL}...")
    response = await asyncio.to_thread(
        requests.post,
        f"{base_url}/api/pull",
        json={"model": OLLAMA_MODEL, "stream": False},
        timeout=(5, OLLAMA_PULL_TIMEOUT_SECONDS),
    )
    response.raise_for_status()
    print(f"Ollama model {OLLAMA_MODEL} is ready.")

@bot.event
async def on_ready():
    """Log the bot identity once Discord finishes connecting."""
    # Tell the type checker that User is filled up at this point
    assert bot.user is not None
    print('------')
    print(f'Logged in as {bot.user} (ID: {bot.user.id})')
    print('------')

async def main():
    """Register the command cog and run the Discord bot."""
    if OLLAMA_AUTO_PULL:
        await pull_ollama_model()
    async with bot:
        bot.add_cog(BardBotClient(bot))
        await bot.start(DISCORD_TOKEN)
        
asyncio.run(main())