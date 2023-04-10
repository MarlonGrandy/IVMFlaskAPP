import pandas as pd
import numpy as np
import openai as ai
from pydub import AudioSegment
from pydub.playback import play
from dotenv import load_dotenv
import os


def combine_audio():
    # Load the audio files
    speaker_sound = AudioSegment.from_file(
        "path/to/speaker_sound.wav", format="wav")
    new_sound = AudioSegment.from_file("path/to/new_sound.wav", format="wav")

    # Adjust the volume of the new sound
    new_sound = new_sound + 6  # Increase the volume by 6 dB

    # Mix the two sounds
    mixed_sound = speaker_sound.overlay(new_sound)
    play(mixed_sound)


def run_pipeline(txt1, txt2):
    load_dotenv()
    ai.api_key = os.getenv('OPEN_AI_KEY')
    env_data = np.random.randint(0, 81, (100, 3))
    env_df = pd.DataFrame(
        env_data, columns=['temperature', 'humidity', 'tide'])
    prompt_dict = {10: "angry", 25: "sad", 40: "disgusted",
                   50: "humorous", 60: "happy", 70: "joyous"}
    rand_idx = np.random.randint(0, env_df.shape[0])
    cur_data = env_df.loc[rand_idx, :]
    mapped = min(prompt_dict.keys(), key=lambda x: abs(
        x - cur_data['temperature']))
    mood = prompt_dict[mapped]
    prompt = f'Write one paragraph interweaving the styles and themes of this text: "{txt1}" and this text: "{txt2}". Use both texts relatively equally. Make the paragraph sound {mood}.'
    response = ai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=256,
    )
    return (response['choices'][0]['text'])


