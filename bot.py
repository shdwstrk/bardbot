import faster_whisper as whisper
import tqdm
import discord
import io
import os
import asyncio
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
import requests

from discord.ext import commands
from pydub import AudioSegment
from typing import Dict

BASE_DIR = Path(__file__).resolve().parent
DISCORD_TOKEN = str(os.getenv("DISCORD_TOKEN", ""))
RECORDINGS_DIR = Path( BASE_DIR / os.getenv("RECORDINGS_DIR", "recordings"))
TRANSCRIPTS_DIR = Path( BASE_DIR / os.getenv("TRANSCRIPTS_DIR", "transcripts"))
OLLAMA_HOST = str(os.getenv("OLLAMA_HOST", "localhost"))
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "360"))
OLLAMA_MODEL = str(os.getenv("OLLAMA_MODEL", "llama3.1:latest"))
if(Path("character_names.json").exists()):
    with open("character_names.json") as f:
        CHARACTER_MAP = json.load(f)
else:
    CHARACTER_MAP = {}
    
class BardBotClient(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.voice_client = None

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
            if self.voice_client.recording:
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
            if self.voice_client.recording:
                await ctx.send('Already recording!')
                print('Failed to start recording: Already recording!')
                return
            self.voice_client.start_recording(
                discord.sinks.WaveSink(),
                self._recording_finished
            )
            await ctx.send('Recording started!')
    
    async def _recording_finished(self, sink):
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
        """Stop recording audio."""
        if not self.voice_client or not self.voice_client.is_connected():
            await ctx.send('I am not in a voice channel!')
            print('Failed to stop recording: Not in a voice channel!')
            return
        if not self.voice_client.recording:
            await ctx.send('No recording in progress.')
            return
        else:
            self.voice_client.stop_recording()
            await ctx.send('Recording stopped!')
    
    @commands.command()
    async def transcribe(self, ctx, session: str = ""):
        """Transcribe audio."""
        await ctx.send('Transcribing audio... This may take a while depending on the length of the recording.')
        if session == "":
            session = os.listdir(RECORDINGS_DIR)[-1]  # Get the latest session
        if not Path(RECORDINGS_DIR / session).exists():
            await ctx.send(f'Session {session} does not exist.')
            print(f'Session {session} does not exist.')
            return
        session_dir = Path(RECORDINGS_DIR / session )
        print('Transcribing audio... This may take a while depending on the length of the recording.')
        model = whisper.WhisperModel("large-v3", device="cuda", compute_type="float16")
        transcript = []
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
        await ctx.send(f'Transcription completed for session {session}. Transcript saved.')
    
    @commands.command()
    async def summarize(self, ctx, session: str = ""):
        """Summarize transcribed audio."""
        await ctx.send('Summarizing audio...')
        print('Summarizing audio...')
        if session == "":
            session = os.listdir(RECORDINGS_DIR)[-1]  # Get the latest session
        transcript_file = Path(TRANSCRIPTS_DIR / f"{session}.txt")
        if not transcript_file.exists():
            await ctx.send(f'Transcript for session {session} does not exist. Please transcribe the session first.')
            print(f'Transcript for session {session} does not exist. Please transcribe the session first.')
            return
        transcript_data = ""
        with open(transcript_file, "r") as transcript:
            for line in transcript:
                item = json.loads(line)
                text = item.get("text", "").strip()
                userid = item.get('user_id')
                user = CHARACTER_MAP.get(userid, userid)
                transcript_data += f"{user} : {text}\n"
        if len(transcript_data) < 10:
            await ctx.send(f'Transcript for session {session} is too short to summarize or is empty.')
            print(f'Transcript for session {session} is too short to summarize or is empty.')
            return
        await ctx.send(f'Summarizing transcript for session {session}...')
        print(f'Summarizing transcript for session {session}...')
        prompt = f"""You are a careful D&D campaign transcript chronicler. Summarize only facts supported by the transcript or relevant notes; do not invent events, entities, items, or outcomes. If a detail is uncertain, say it is not established. Write an evocative but factual campaign journal recap with sections for Session Summary, Key Moments, Characters and Places, Decisions and Consequences, Loot and Progression, Open Threats, and Next Session Actions.\n\nTranscript:\n{transcript_data}"""
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": 65000,
                "temp": 0.2
            }
        }
        try:
            response = await asyncio.to_thread(
                requests.post,
                f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate",
                json=payload,
                timeout=OLLAMA_TIMEOUT_SECONDS,
            )
        except requests.exceptions.ReadTimeout:
            print("Ollama timed out after %.0f seconds while summarizing session %s", OLLAMA_TIMEOUT_SECONDS, session)
            await ctx.send(
                f"Ollama did not finish summarizing within {OLLAMA_TIMEOUT_SECONDS:.0f} seconds. "
                "Try again, increase OLLAMA_TIMEOUT_SECONDS, or use a smaller model/context."
            )
            return
        response.raise_for_status()
        summary = (response.json().get("response") or "").strip()
        summary_file = TRANSCRIPTS_DIR / f"{session}_summary.txt"
        with open(summary_file, "w") as f:
            f.write(summary)
            print(f'Summary for session {session} saved to {summary_file}.')
        
        await ctx.send(file=discord.File(summary_file))
        
    @commands.command()
    async def list_sessions(self, ctx):
        """List all recorded sessions."""
        sessions = sorted(os.listdir(RECORDINGS_DIR))
        if not sessions:
            await ctx.send('No recorded sessions found.')
            print('No recorded sessions found.')
            return
        session_list = "\n".join(sessions)
        await ctx.send(f'Available recorded sessions:\n{session_list}')
        print("Session list sent to channel.")
        
    
        
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    # Tell the type checker that User is filled up at this point
    assert bot.user is not None
    print(f'Logged in as {bot.user} (ID: {bot.user.id})')
    print('------')

async def main():
    async with bot:
        bot.add_cog(BardBotClient(bot))
        await bot.start(DISCORD_TOKEN)
        
asyncio.run(main())