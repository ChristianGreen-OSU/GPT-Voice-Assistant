import os
import openai
import pyaudio
from google.cloud import speech_v1p1beta1 as speech
from google.cloud.speech_v1p1beta1.types import RecognitionConfig, StreamingRecognitionConfig, StreamingRecognizeRequest
from config import OPEN_AI_API_KEY, XI_API_KEY
from elevenlabs import voices, generate, play, set_api_key


# Set the environment variable for the JSON key file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\Users\green\credentials\empirical-lens-385900-77667f075e80.json'
if OPEN_AI_API_KEY is None:
    raise ValueError("The OPENAI_API_KEY environment variable is not set")
openai.api_key = OPEN_AI_API_KEY
set_api_key(XI_API_KEY)

voices = voices()

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms chunks
RECORD_SECONDS = 10  # Duration of the recording in seconds

def transcribe_streaming():
    client = speech.SpeechClient()

    config = RecognitionConfig(
        encoding=RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code='en-US'
    )
    streaming_config = StreamingRecognitionConfig(
        config=config,
        interim_results=True
    )

    audio_interface = pyaudio.PyAudio()
    audio_stream = audio_interface.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    print("Start speaking...")
    requests = (
        StreamingRecognizeRequest(audio_content=audio_stream.read(CHUNK))
        for _ in range(int(RATE / CHUNK * RECORD_SECONDS))
    )

    responses = client.streaming_recognize(streaming_config, requests)

    for response in responses:
        if not response.results:
            continue

        result = response.results[0]

        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript

        if result.is_final:
            print('Final transcript: {}'.format(transcript))
            return transcript
        else:
            print('Intermediate transcript: {}'.format(transcript))

    audio_stream.stop_stream()
    audio_stream.close()
    audio_interface.terminate()

def gpt_takes_text_input(prompt, model='text-davinci-003', max_tokens=100):
    print('Text input passed to gpt_takes_text_input: {}'.format(prompt))
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=0.7,
    )

    message = response.choices[0].text.strip()
    return message

def play_response(response):
    audio = generate(
        text=response,
        voice=voices[-1],
        model="eleven_monolingual_v1",
    )   
    play(audio)

if __name__ == '__main__':
    mic_input = transcribe_streaming()
    gpt_response = gpt_takes_text_input(mic_input)
    print('GPT response: {}'.format(gpt_response))
    # print(voices)
    play_response(gpt_response)
