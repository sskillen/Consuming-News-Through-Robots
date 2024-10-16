import speech_recognition as sr
import pyttsx3

# Initialize Recognizer and pyttsx3 Engine
recognizer = sr.Recognizer()
engine = pyttsx3.init()
# Test speech synthesis
#def test_speech():
    #engine.say("Hello! Testing audio output.")
    #engine.runAndWait()

# Call the test speech function
#test_speech()
# Use the device index directly for your Bluetooth microphone
device_index = 1  # 'Microphone (Jabra Link 370)' has device_index 1

# List Voices
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # Change the index if needed

# Pre-defined Newscaster responses
responses = {
    "hallo": "Hoi, ik ben Jip, je virtuele assistent, je cloud-based AI. Als je naar een nieuwsbriefing wilt luisteren, zeg dan: Jip, speel het nieuws",
    "jip speel het nieuws": "Uit het N.O.S. Journaal: Rechters laten zich beïnvloeden door het opleidingsniveau en de achtergrond van verdachten. Dat blijkt uit onderzoek van N.O.S. op 3 en Investico. Zonder vervolgopleiding en met een migratieachtergrond krijgen verdachten bijna drie keer zo vaak een celstraf in vergelijking met verdachten zonder migratieachtergrond en met een h.b.o.-of w.o. opleiding. Studenten met een beperking maken steeds vaker gebruik van een speciale aanvullende studiebeurs. Er zijn er nu ruim 9.000, meldt belangenorganisatie Ieder(in). De twintigjarige Fleur is slechtziend en is daardoor bijvoorbeeld meer tijd kwijt aan het lezen van studieboeken en heeft geen tijd en energie om naast haar studie te werken. Ze is blij met de toeslag... Het geeft mij heel veel rust en nu kan ik ook echt de focus leggen op mijn school, zonder dat ik stress heb over geld of werk of zoiets. En ja, ik heb eigenlijk mijn leven terug, want nu kan ik weer gewoon na school iets leuks gaan doen... De V.V.D. wil dat er een kiesdrempel komt voor de Tweede Kamer om versplintering in het parlement tegen te gaan. Op dit moment zitten er 15 partijen in de Kamer. Door alle afsplitsingen was dat nog veel meer in de vorige Kamer. Een kiesdrempel kan daar wat aan doen, zegt VVD-Kamerlid. Telkens, een kiesdrempel zorgt ervoor dat je een minimaal aantal zetels moet halen om in het parlement te komen. Dus door een kiesdrempel krijg je effectief minder politieke partijen in de Tweede Kamer, zodat de politiek zich ook meer kan gaan bezighouden met de inhoud, in plaats van continu met elkaar. Als het aan de VVD ligt, kan een partij pas in de Kamer komen als die minimaal 3 zetels haalt... De EU-landen zijn het eens geworden over een gezamenlijk standpunt voor de klimaattop volgende maand. De landen roepen op om de opwarming van de aarde te beperken tot maximaal 1,5 graad. Nederland wil kernenergie noemen als middel om de klimaatdoelen te halen, maar daarover werden de EU-landen het niet eens... Bijna één op de vijf Nederlanders voelt zich niet heteroseksueel. Dat is een inschatting van het CBS op basis van een groot bevolkingsonderzoek. Zo'n 2,7 miljoen mensen in Nederland noemen zich lesbisch, homoseksueel, bi+, transgender, queer, intersekse of aseksueel. Het is voor het eerst dat het Centraal Bureau voor de Statistiek met cijfers komt over deze groepen." 
    }

#def speak(text):
   ## engine.say(text)
   # engine.runAndWait()

def speak(text):
    # Set the output device for the pyttsx3 engine
    engine.setProperty('rate', 150)  # You can adjust the speech rate if needed
    engine.setProperty('volume', 1)   # Set volume level (0.0 to 1.0)
    engine.say(text)
    engine.runAndWait()


def listen():
    # Use the specified microphone device by index
    with sr.Microphone(device_index=device_index) as source:
        print("Listening. . .")
        audio = recognizer.listen(source)
        try:
            command = recognizer.recognize_google(audio, language="nl-NL").lower()
            print(f"You said: {command}")
            return command
        except sr.UnknownValueError:
            print("Sorry, I did not understand that.")
            return None
        except sr.RequestError:
            print("Could not request results from Google Speech Recognition service.")
            return None

# Main Loop
while True:
    command = listen()
    
    if command:
        print(f"Command recognized: {command}")
        
        if command in responses:
            print("Speaking response . . .")
            speak(responses[command])

            # check if news briefing response is spoken
            
            if command == "jip speel het nieuws":
                

            # Speak the ending message and then break out of the loop
                speak("Dat is alles voor nu.")
                break # exit loop
        
        else:
            # Response for unrecognized commands
            unrecognized_message = "Sorry, dat heb ik niet begrepen. Probeer het nog eens en spreek luid en duidelijk."
            print("Speaking unrecognized response . . .")
            speak(unrecognized_message)

