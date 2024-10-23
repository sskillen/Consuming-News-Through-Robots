import speech_recognition as sr
import pyttsx3
import random

# Initialize Recognizer and pyttsx3 Engine
recognizer = sr.Recognizer()
engine = pyttsx3.init()

# Use the device index directly for your Bluetooth microphone
device_index = 1  # 'Microphone (Jabra Link 370)' has device_index 1

# List Voices
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # Change the index if needed

# Pre-defined Newscaster responses
responses = {
     "nieuws": [
        "Rechters laten zich beïnvloeden door het opleidingsniveau en de achtergrond van verdachten. Dat blijkt uit onderzoek van N.O.S. op 3 en Investico. Zonder vervolgopleiding en met een migratieachtergrond krijgen verdachten bijna drie keer zo vaak een celstraf in vergelijking met verdachten zonder migratieachtergrond en met een h.b.o.-of w.o. opleiding... ", 
        "Studenten met een beperking maken steeds vaker gebruik van een speciale aanvullende studiebeurs. Er zijn er nu ruim 9.000, meldt belangenorganisatie Ieder(in). De twintigjarige Fleur is slechtziend en is daardoor bijvoorbeeld meer tijd kwijt aan het lezen van studieboeken en heeft geen tijd en energie om naast haar studie te werken. Ze is blij met de toeslag... Het geeft mij heel veel rust en nu kan ik ook echt de focus leggen op mijn school, zonder dat ik stress heb over geld of werk of zoiets. En ja, ik heb eigenlijk mijn leven terug, want nu kan ik weer gewoon na school iets leuks gaan doen... ", 
        "De V.V.D. wil dat er een kiesdrempel komt voor de Tweede Kamer om versplintering in het parlement tegen te gaan. Op dit moment zitten er 15 partijen in de Kamer. Door alle afsplitsingen was dat nog veel meer in de vorige Kamer. Een kiesdrempel kan daar wat aan doen, zegt VVD-Kamerlid. Telkens, een kiesdrempel zorgt ervoor dat je een minimaal aantal zetels moet halen om in het parlement te komen. Dus door een kiesdrempel krijg je effectief minder politieke partijen in de Tweede Kamer, zodat de politiek zich ook meer kan gaan bezighouden met de inhoud, in plaats van continu met elkaar. Als het aan de VVD ligt, kan een partij pas in de Kamer komen als die minimaal 3 zetels haalt...", 
        "Bijna één op de vijf Nederlanders voelt zich niet heteroseksueel. Dat is een inschatting van het CBS op basis van een groot bevolkingsonderzoek. Zo'n 2,7 miljoen mensen in Nederland noemen zich lesbisch, homoseksueel, bi+, transgender, queer, intersekse of aseksueel. Het is voor het eerst dat het Centraal Bureau voor de Statistiek met cijfers komt over deze groepen."
        "De EU-landen zijn het eens geworden over een gezamenlijk standpunt voor de klimaattop volgende maand. De landen roepen op om de opwarming van de aarde te beperken tot maximaal 1,5 graad. Nederland wil kernenergie noemen als middel om de klimaatdoelen te halen, maar daarover werden de EU-landen het niet eens... "]
}
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

# Introduction
speak("Hoi, ik ben Jip, je virtuele assistent, je cloud-based AI. Als je naar een nieuwsbriefing wilt luisteren, zeg dan: Jip, speel het nieuws.")

# Main Loop
while True:
    command = listen()
    
    if command:
        print(f"Command recognized: {command}")

        # Check for specific keywords and respond accordingly
        if "nieuws" in command:
            print("Speaking news response. . .")
            # Announce the source before the news items
            speak("Uit het N.O.S. Journaal:")
            # Randomly select 3 news items
            selected_news = random.sample(responses["nieuws"], 3)
            # Print the selected news items to the terminal
            print("Geselecteerde nieuwsitems:")
            for news_item in selected_news:
                print(news_item)
            # Speak each selected news item
            for news_item in selected_news:
                speak(news_item)
            speak("Dat is alles voor nu.")
            break  # exit loop after speaking the news

        else:
            # Only speak this if the command doesn't match any predefined response
            unrecognized_message = "Sorry, dat heb ik niet begrepen. Probeer het nog eens en spreek luid en duidelijk."
            print("Speaking unrecognized response. . .")
            speak(unrecognized_message)