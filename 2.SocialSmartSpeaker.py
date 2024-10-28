import speech_recognition as sr
import pyttsx3
import random
import time

# Initialize Recognizer and pyttsx3 Engine
recognizer = sr.Recognizer()
engine = pyttsx3.init()

# Use the device index directly for your Bluetooth microphone
device_index = 1  # 'Microphone (Jabra Link 370)' has device_index 1

# List Voices
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # Change the index if needed

# Pre-defined Newscaster responses (5 news items)
news_content = [
    "Rechters laten zich beïnvloeden door het opleidingsniveau en de achtergrond van verdachten. Dat blijkt uit onderzoek van N.O.S. op 3 en Investico. Zonder vervolgopleiding en met een migratieachtergrond krijgen verdachten bijna drie keer zo vaak een celstraf in vergelijking met verdachten zonder migratieachtergrond en met een h.b.o.-of w.o. opleiding...",
    "Studenten met een beperking maken steeds vaker gebruik van een speciale aanvullende studiebeurs. Er zijn er nu ruim 9.000, meldt belangenorganisatie Ieder(in). De twintigjarige Fleur is slechtziend en is daardoor bijvoorbeeld meer tijd kwijt aan het lezen van studieboeken en heeft geen tijd en energie om naast haar studie te werken. Ze is blij met de toeslag... Het geeft mij heel veel rust en nu kan ik ook echt de focus leggen op mijn school, zonder dat ik stress heb over geld of werk of zoiets. En ja, ik heb eigenlijk mijn leven terug, want nu kan ik weer gewoon na school iets leuks gaan doen...",
    "De V.V.D. wil dat er een kiesdrempel komt voor de Tweede Kamer om versplintering in het parlement tegen te gaan. Op dit moment zitten er 15 partijen in de Kamer. Door alle afsplitsingen was dat nog veel meer in de vorige Kamer. Een kiesdrempel kan daar wat aan doen, zegt VVD-Kamerlid. Telkens, een kiesdrempel zorgt ervoor dat je een minimaal aantal zetels moet halen om in het parlement te komen. Dus door een kiesdrempel krijg je effectief minder politieke partijen in de Tweede Kamer, zodat de politiek zich ook meer kan gaan bezighouden met de inhoud, in plaats van continu met elkaar. Als het aan de VVD ligt, kan een partij pas in de Kamer komen als die minimaal 3 zetels haalt...",
    "Bijna één op de vijf Nederlanders voelt zich niet heteroseksueel. Dat is een inschatting van het CBS op basis van een groot bevolkingsonderzoek. Zo'n 2,7 miljoen mensen in Nederland noemen zich lesbisch, homoseksueel, bi+, transgender, queer, intersekse of aseksueel. Het is voor het eerst dat het Centraal Bureau voor de Statistiek met cijfers komt over deze groepen...",
    "De EU-landen zijn het eens geworden over een gezamenlijk standpunt voor de klimaattop volgende maand. De landen roepen op om de opwarming van de aarde te beperken tot maximaal 1,5 graad. Nederland wil kernenergie noemen als middel om de klimaatdoelen te halen, maar daarover werden de EU-landen het niet eens..."
]

def speak(text):
    # Set the output device for the pyttsx3 engine
    engine.setProperty('rate', 150)  # Adjust the speech rate if needed
    engine.setProperty('volume', 1)  # Set volume level (0.0 to 1.0)
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

# Initial interaction
speak("Hallo! Ik ben Jip, hoe heet jij?")  # "Hello! I am Jip. What's your name?"

time.sleep(3)  # Wait for 3 seconds
# Ask how many events they would like to hear
speak("Ik ben blij dat je vandaag de tijd hebt genomen om hier te komen. Ik vind het erg leuk om nieuwe mensen te ontmoeten en actuele gebeurtenissen in hun omgeving en in de wereld te bespreken. Vandaag zal ik je een paar nieuwsberichten laten horen. Zou je willen horen over drie actuele gebeurtenissen, twee actuele gebeurtenissen, of één actuele gebeurtenis?")  # "Would you like to hear three, two, or one current event?"

# Loop until a valid number of events is heard
# Loop until a valid number of events is heard
event_choice = None
while not event_choice:
    event_choice = listen()
    if event_choice:
        # Check for both literal numbers and spelled-out numbers
        if "drie" in event_choice or "3" in event_choice:
            num_events = 3
        elif "twee" in event_choice or "2" in event_choice:
            num_events = 2
        elif "één" in event_choice or "een" in event_choice or "1" in event_choice:
            num_events = 1
        else:
            speak("Sorry, dat is geen geldige keuze. Zeg alsjeblieft één, twee of drie.")  # "Sorry, that's not a valid option. Please say one, two, or three."
            event_choice = None
    else:
        speak("Sorry, ik heb je niet goed verstaan. Probeer het opnieuw.")  # "Sorry, I didn't catch that. Please try again."



# Select and speak the requested number of events
selected_news = random.sample(news_content, num_events)

print(selected_news)

for i, news_item in enumerate(selected_news, 1):
    speak("Uit het N.O.S. Journaal: " + news_item)

# After speaking the selected news content, ask for the user's feedback
speak("Dat is alle informatie die ik vandaag heb. Wat vond je ervan?")  # "That's all the information I have for today. What did you think?"

time.sleep(3)  # Wait for 3 seconds

speak("Oh, interessant. Ik wil vandaag niet meer van je tijd in beslag nemen, dus we ronden het af. Bedankt voor het gesprek en ik hoop je snel weer te zien. Tot ziens en een fijne rest van je dag!")  # "Oh, interesting. I don't want to take more of your time today, so let's wrap it up. Thanks for the conversation and I hope to see you soon. Goodbye and have a nice day!"

