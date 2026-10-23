import wave

import discord


class DaveWaveSink(discord.sinks.WaveSink):
    def write(self, data, user):
        user_id = getattr(user, "id", user)
        super().write(data, user_id)

    def format_audio(self, audio):
        pcm = audio.file.getvalue()
        audio.file.seek(0)
        audio.file.truncate()

        with wave.open(audio.file, "wb") as wav_file:
            wav_file.setnchannels(2)
            wav_file.setsampwidth(2)
            wav_file.setframerate(48000)
            wav_file.writeframes(pcm)

        audio.file.seek(0)
        audio.on_format(self.encoding)
