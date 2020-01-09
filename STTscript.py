import json
from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import ApiException
from os.path import join, dirname
import os
import re

authenticator = IAMAuthenticator('ChhShStDFkt_ZS7acxqu4DFKp_cWMaTFEYMB3IqvkDYN')
speech_to_text = SpeechToTextV1(
    authenticator=authenticator
)

speech_to_text.set_service_url('https://stream.watsonplatform.net/speech-to-text/api')
speech_to_text.set_default_headers({'x-watson-learning-opt-out': "true"})
# speech_to_text.set_disable_ssl_verification(True)

directory_in_str = join(dirname(__file__), './audiofiles')
directory = os.fsencode(directory_in_str)

all_transcripts = []


with open( join(dirname(__file__), './audiotranscriptions', 'alltranscriptions.csv'),
             "w") as write_file:
    write_file.write('Audio File,Untrained Model,Trained Model'+'\n')

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".mp3"): 
            with open(join(dirname(__file__), './audiofiles', filename),
                    'rb') as audio_file1:
                try:
                    print('untrained model')
                    untrained_speech_recognition_results = speech_to_text.recognize(
                        audio=audio_file1,
                        model='en-GB_NarrowbandModel',
                        content_type='audio/mp3',
                        word_alternatives_threshold=0.9,
                    ).get_result()
                    full_untrained_transcript = ''
                    for sentence in untrained_speech_recognition_results['results']:
                        full_untrained_transcript += str(sentence['alternatives'][0]['transcript']) + '. '
                    full_untrained_transcript = full_untrained_transcript.replace('%', '').replace('HESITATION', '')
                    print(filename + ': ' + full_untrained_transcript)

                except ApiException as ex:
                    print("Method failed with status code " + str(ex.code) + ": " + ex.message)

            with open(join(dirname(__file__), './audiofiles', filename),
                    'rb') as audio_file2:
                try:
                    print('trained model')
                    trained_speech_recognition_results = speech_to_text.recognize(
                        audio=audio_file2,
                        model='en-GB_NarrowbandModel',
                        content_type='audio/mp3',
                        word_alternatives_threshold=0.9,
                        acoustic_customization_id='88ebe702-2ac6-42b8-b2b4-bdb947b10ecf',
                        language_customization_id='ac03b998-ec3b-4d3d-8349-5f8789af5b3e',
                    ).get_result()
                    full_trained_transcript = ''
                    for sentence in trained_speech_recognition_results['results']:
                        full_trained_transcript += str(sentence['alternatives'][0]['transcript']) + '. '
                    full_trained_transcript = full_trained_transcript.replace('%', '').replace('HESITATION', '')
                    print(filename + ': ' + full_trained_transcript)

                except ApiException as ex:
                    print("Method failed with status code " + str(ex.code) + ": " + ex.message)

            write_file.write(filename + ',' + full_untrained_transcript + ',' + full_trained_transcript + '\n')

        else:
            continue

