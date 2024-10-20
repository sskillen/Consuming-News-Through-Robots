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

# Pre-defined Newscaster responses (5 news items)
options = ["Hoe de gevangenisstraffen van rechters worden beïnvloed door de achtergrond van verdachten.",
           "Een speciale aanvullende studiebeurs voor gehandicapte studenten.", 
           "De potentiële voordelen van een kiesdrempel in het parlement.", 
           "Recente statistieken over de seksuele geaardheid van individuen in Nederland",
           "De EU-klimaattop."
    
]
news_content = ["Rechters laten zich beïnvloeden door het opleidingsniveau en de achtergrond van verdachten. Dat blijkt uit onderzoek van N.O.S. op 3 en Investico. Zonder vervolgopleiding en met een migratieachtergrond krijgen verdachten bijna drie keer zo vaak een celstraf in vergelijking met verdachten zonder migratieachtergrond en met een h.b.o.-of w.o. opleiding... ", 
        "Studenten met een beperking maken steeds vaker gebruik van een speciale aanvullende studiebeurs. Er zijn er nu ruim 9.000, meldt belangenorganisatie Ieder(in). De twintigjarige Fleur is slechtziend en is daardoor bijvoorbeeld meer tijd kwijt aan het lezen van studieboeken en heeft geen tijd en energie om naast haar studie te werken. Ze is blij met de toeslag... Het geeft mij heel veel rust en nu kan ik ook echt de focus leggen op mijn school, zonder dat ik stress heb over geld of werk of zoiets. En ja, ik heb eigenlijk mijn leven terug, want nu kan ik weer gewoon na school iets leuks gaan doen... ", 
        "De V.V.D. wil dat er een kiesdrempel komt voor de Tweede Kamer om versplintering in het parlement tegen te gaan. Op dit moment zitten er 15 partijen in de Kamer. Door alle afsplitsingen was dat nog veel meer in de vorige Kamer. Een kiesdrempel kan daar wat aan doen, zegt VVD-Kamerlid. Telkens, een kiesdrempel zorgt ervoor dat je een minimaal aantal zetels moet halen om in het parlement te komen. Dus door een kiesdrempel krijg je effectief minder politieke partijen in de Tweede Kamer, zodat de politiek zich ook meer kan gaan bezighouden met de inhoud, in plaats van continu met elkaar. Als het aan de VVD ligt, kan een partij pas in de Kamer komen als die minimaal 3 zetels haalt...", 
        "Bijna één op de vijf Nederlanders voelt zich niet heteroseksueel. Dat is een inschatting van het CBS op basis van een groot bevolkingsonderzoek. Zo'n 2,7 miljoen mensen in Nederland noemen zich lesbisch, homoseksueel, bi+, transgender, queer, intersekse of aseksueel. Het is voor het eerst dat het Centraal Bureau voor de Statistiek met cijfers komt over deze groepen.",
        "De EU-landen zijn het eens geworden over een gezamenlijk standpunt voor de klimaattop volgende maand. De landen roepen op om de opwarming van de aarde te beperken tot maximaal 1,5 graad. Nederland wil kernenergie noemen als middel om de klimaatdoelen te halen, maar daarover werden de EU-landen het niet eens... "]


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
speak("Hallo! Ik ben Jip, hoe heet jij?")  # "What's your name?"

# Loop until a valid name is heard
name = None
while not name:
    name = listen()
    if name:
        print(f"User's name: {name}")
        speak(f"Bedankt voor het delen van uw naam.")
    else:
        speak("Sorry, ik heb je niet goed verstaan. Probeer het alsjeblieft opnieuw.")  # "Sorry, I didn't catch that. Please try again."

    # Randomly select 3 news items from the list
speak("Ik ben blij dat je vandaag de tijd hebt genomen om hier te komen. Ik vind het erg leuk om nieuwe mensen te ontmoeten en actuele gebeurtenissen in hun omgeving en in de wereld te bespreken. Ik zal een aantal mogelijke onderwerpen van N.O.S. geven, je kunt zelf beslissen of je wilt dat ik je er meer over vertel of niet. Je kunt nieuws horen over het volgende... ")
    
selected_options = random.sample(options, 3)
option_mapping = {}  # Dictionary to store option numbers and their corresponding news items

    # Speak each selected news item and associate it with a spelled-out option number in Dutch
dutch_numbers = {1: "één", 2: "twee", 3: "drie"}  # Mapping of numbers to Dutch words
for i, news_item in enumerate(selected_options, 1):
    speak(f"Optie {dutch_numbers[i]}: {news_item}")
    option_mapping[i] = news_item  # Store the mapping of option number to news item

    # Ask the user to choose an option
speak("Erg interessante lijst, nietwaar?! Als je er zin in hebt, vertel me dan welk nieuwsverhaal je graag meer wilt horen door optie één, optie twee of optie drie te noemen.")

    # Optionally, print the option mapping for debugging purposes
print(option_mapping)
    
while True:
    # Listen for the user's choice of option
        user_choice = listen()

    # Check if the user mentioned a valid option
        if user_choice:
            if "optie één" in user_choice:
                selected_news = option_mapping[1]
                speak(news_content[options.index(selected_news)])  # Use the index of the selected news title to get the full news piece
            elif "optie twee" in user_choice:
                selected_news = option_mapping[2]
                speak(news_content[options.index(selected_news)])
            elif "optie drie" in user_choice:
                selected_news = option_mapping[3]
                speak(news_content[options.index(selected_news)])
            else:
                speak("Sorry, dat is geen geldige optie.")  # "Sorry, that's not a valid option."

        # After speaking the selected news content, ask for the user's feedback
            speak("Dat is alle informatie die ik heb voor dat onderwerp. Wat vond je ervan?")  # "That's all the information I have on that topic. What did you think?"

        # Wait for user's response
            feedback = listen()

        # After listening to feedback, give a closing response
            if feedback:
                speak("Oh, intereessant. . . Ik wil vandaag niet meer van je tijd in beslag nemen, dus we ronden het af. Bedankt voor het gesprek en ik hoop je snel weer te zien. Tot ziens en een fijne rest van je dag!")
                break #exit loop after closing response  
            else:
                speak("Sorry, ik heb je niet goed verstaan.")  # "Sorry, I didn't catch that."
        else:
            speak("Sorry, ik heb je niet goed verstaan.")  # "Sorry, I didn't catch that."