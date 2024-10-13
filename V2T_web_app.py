
############### Import Libraries ###############
import streamlit as st
import speech_recognition as sr
import whisper 
from pydub import AudioSegment
from textblob import TextBlob 

# Load the whisper medium model
model = whisper.load_model("small")

# Create recognizer instance
recognizer = sr.Recognizer()

# Initialize variables to store audio file or recorded audio path
audio_file_path = None

page_bg_img = '''
<style>
.st-emotion-cache-1r4qj8v {
background-image: url("https://nlpcloud.io/assets/images/dark-background.jpg");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

############### Function ###############
def audio_file_upload():
    uploaded_file = st.file_uploader("Upload an audio file", type=['wav', 'mp3'])

    if uploaded_file is not None:
        # Load the uploaded audio file
        audio = AudioSegment.from_file(uploaded_file)
        
        # Save the file in WAV format
        audio_file_path = "uploaded_audio.wav"
        audio.export(audio_file_path, format="wav")

        # Display the audio player to play the recorded audio
        st.audio(audio_file_path, format='audio/wav')
        
        # Transcribe the uploaded audio file
        st.write("Transcribing uploaded audio...")
        result = model.transcribe(audio_file_path, fp16=False)

        # Detect and display the language
        st.session_state.detected_language = result.get("language", "unknown")
        st.write(f"Detected Language: {st.session_state.detected_language}")
                
        # Save the transcription
        st.session_state.transcribed_text = result["text"]  # Store in session state
        st.write("Transcription:")
        st.write(st.session_state.transcribed_text)

############### Function to handle Voice Input ###############
def voice_input():
    # Button to start recording
    if st.button("Start Recording"):
        with sr.Microphone() as source:
            # Record voice input from the microphone
            st.write("Recording... Speak now!")
            audio_data = recognizer.listen(source)
            st.write("Recording has finished after user stopped speaking.")

        # Save the recorded audio to a WAV file
        audio_file_path = "recorded_audio.wav"
        with open(audio_file_path, "wb") as f:
            f.write(audio_data.get_wav_data())
        
        # Transcribe the recorded audio using Whisper
        st.write("Transcribing recorded audio...")
        result = model.transcribe(audio_file_path, fp16=False)

        # Detect and display the language
        st.session_state.detected_language = result.get("language", "unknown")
        st.write(f"Detected Language: {st.session_state.detected_language}")

        # Save the transcription and language in session state
        st.session_state.transcribed_text = result["text"]
        st.write("Transcription:")
        st.write(st.session_state.transcribed_text)

############### Sentiment Analysis Function ###############
def perform_sentiment_analysis(text):
    # Perform sentiment analysis using TextBlob
    blob = TextBlob(text)
    sentiment = blob.sentiment

    # Extract polarity
    polarity = sentiment.polarity

    # Determine the sentiment category based on polarity score
    if polarity > 0:
        sentiment_type = "Positive"
        st.success(f"The sentiment of this transcribed text is {sentiment_type} with a polarity score of {polarity:.2f}.")
    elif polarity < 0:
        sentiment_type = "Negative"
        st.error(f"The sentiment of this transcribed text is {sentiment_type} with a polarity score of {polarity:.2f}.")
    else:
        sentiment_type = "Neutral"
        st.success(f"The sentiment of this transcribed text is {sentiment_type} with a polarity score of {polarity:.2f}.")


############### Main Page Function ###############

# Main Page
def main():

    # Initialize session state for transcribed text
    if "transcribed_text" not in st.session_state:
        st.session_state.transcribed_text = None
    if 'detected_language' not in st.session_state:
        st.session_state.detected_language = None
        
    st.markdown("""
            <style> 
            h1{
                color: white;
                } 
            .st-emotion-cache-1vbkxwb p{
                color: black;
                } 
            p, ol, ul, dl{
                color: white;
                }
            .st-emotion-cache-1vt4y43{
                background-color: #b77fff;
                }
            .css-1e5imcs{
                color: white;
                }
            .css-1ekf893{
                color: white;
                font-weight: bold;
                font-size:  16px;
                }
            .css-qbe2hs{
                color: black;
                font-weight: bold;
                }
            .css-glyadz{
                color: white;
                font-size: 24px;
                }
            .code{
                color: black;
                }
            .css-10trblm{
                font-weight: bold;
                color: white;
                }
            .st-cd {
                background-color: rgb(215 56 50);
                }
            hr{
                border-bottom: 2px solid white;
                }
            code{
                font-weight: bolder;
                }
            .css-h9oeas{
                color: white;
                }
            .css-120qjcf, .css-14n4bfl:hover{
                color: black;
                }     
            .css-gh49vm{
                background-color: lightgray;
                }
            h3 {
                color: rgb(255 255 255);
                }
            
                                                </style>"""
            , unsafe_allow_html=True)    
    
    global audio_file_path, audio_data, transcribed_text
    
    #Skip line
    st.write("""""") 

    # Title
    st.title("Voice To Text Web Application")

    st.write("""
        This web application has been developed for converting voice to text. 
        You can either upload an audio file or use voice input from your microphone.
    """)

    st.write('---')

    # Option between audio file uploader & voice input recorder
    option = st.radio("Select input method:", ('Upload an audio file', 'Record voice input'))

    if option == 'Upload an audio file':
        audio_file_upload()
    else:
        voice_input()

    st.write('---')
    st.write("You can also check the sentiment of the transcribed text.")
    
    # Show sentiment analysis button
    if st.button('Check Sentiment'):
        # Check if transcription exists
        if st.session_state.transcribed_text:
            # Verify the detected language
            if st.session_state.detected_language == 'en':
                # Perform sentiment analysis if the detected language is English
                perform_sentiment_analysis(st.session_state.transcribed_text)
            else:
                st.warning(f"Sentiment analysis is supported for English. Detected language: {st.session_state.detected_language}.")
        else:
            st.warning("No transcribed text found! Please upload an audio file or record a voice input first.")

    #Skip line
    st.write("""""") 


############### Run the Main Function ###############
if __name__ == "__main__":
    main()




## nothing else so do the good design and deploy

    
    