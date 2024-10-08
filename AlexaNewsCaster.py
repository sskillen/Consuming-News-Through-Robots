import speech_recognition as sr
import pyttsx3
import pyaudio




# Initialize
recognizer = sr.Recognizer()
engine = pyttsx3.init()

#List Voices
voices = engine.getProperty('voices')

# Set a specific voice (for example, the first voice)
engine.setProperty('voice', voices[1].id)  # Change the index to select a different voice

#Pre-defined Newscaster responses
responses = {
    "goedendag": "Hoi, Ik ben Alexa, uw virtuelle secretaress. Zou je naar de nieuws willen luisteren? Zei Alexa, geef mij de nieuws",
    "alexa speel mijn nieuwsbrief": "Here's the latest from your flash briefing. From NOS: twee uur precies van der liende met het NOS Journaal Greenpeace mag van de rechtbank doorgaan met een rechtszaak tegen de Nederlandse overheid de milieuorganisatie wil dat Nederland meer doet om Bonaire te beschermen tegen de gevolgen van klimaatverandering ook moet Nederland meer doen om de uitstoot van broeikasgassen te verminderen begin dit jaar stapte 8 bewoners van Bonaire Samen met Greenpeace naar de rechtbank die 8 bewoners mogen niet meedoen Omdat ze volgens de rechter niet goed konden onderbouwen Waarom zij een eigen rechtstreeks belang zouden hebben Rusland is begonnen met 1 grote aanval op een strategische plaats In de donbas Oekraïense troepen hebben sinds het begin van de oorlog stand kunnen houden in het stadje voelen dag maar nu dreigen ze omsingeld te worden Rusland heeft zo'n 80% van de donbas onder controle president Zelenski is momenteel bij de Verenigde Naties waar hij met wereldleiders spreekt morgen wil hij president Biden zijn overwinnings plan presenteren In Amsterdam zijn de afgelopen maanden 40.000 boetes uitgedeeld aan automobilisten die zich niet aan de nieuwe snelheidslimiet hielden sinds december mag er niet harder dan 30 km per uur worden gereden na een wenperiode zijn sinds kort flitspalen geplaatst de Nederlandse voetbalvrouwen oefenen volgende maand tegen Indonesië en Denemarken zijn de eerste oefenduels richting het EK voetbal volgend jaar zomer bij het Indonesische mannenelftal spelen veel in Nederland geboren spelers ook bij de selectie van de vrouwen wordt daar nu naar gekeken volgens de baas van de Indonesische voetbalbond wordt er binnen afzienbare tijd ook tussen de mannenteams geoefend het weer vanuit het westen komen er nieuwe buien aan bij zon 16 graden tijdens de regen ligt de temperatuur nog wat lager vanavond volgt vanuit het zuidwesten een groot regengebied dan gaat het lange tijd regenen morgen wordt het onstuimig met buien en veel wind dan is het iets warmer bij een graad of 18 dit was het NOS Journaal. That's all for now.",
    "alexa wat zijn de nieuws": "Here's the latest from your flash briefing. From NOS: twee uur precies van der liende met het NOS Journaal Greenpeace mag van de rechtbank doorgaan met een rechtszaak tegen de Nederlandse overheid de milieuorganisatie wil dat Nederland meer doet om Bonaire te beschermen tegen de gevolgen van klimaatverandering ook moet Nederland meer doen om de uitstoot van broeikasgassen te verminderen begin dit jaar stapte 8 bewoners van Bonaire Samen met Greenpeace naar de rechtbank die 8 bewoners mogen niet meedoen Omdat ze volgens de rechter niet goed konden onderbouwen Waarom zij een eigen rechtstreeks belang zouden hebben Rusland is begonnen met 1 grote aanval op een strategische plaats In de donbas Oekraïense troepen hebben sinds het begin van de oorlog stand kunnen houden in het stadje voelen dag maar nu dreigen ze omsingeld te worden Rusland heeft zo'n 80% van de donbas onder controle president Zelenski is momenteel bij de Verenigde Naties waar hij met wereldleiders spreekt morgen wil hij president Biden zijn overwinnings plan presenteren In Amsterdam zijn de afgelopen maanden 40.000 boetes uitgedeeld aan automobilisten die zich niet aan de nieuwe snelheidslimiet hielden sinds december mag er niet harder dan 30 km per uur worden gereden na een wenperiode zijn sinds kort flitspalen geplaatst de Nederlandse voetbalvrouwen oefenen volgende maand tegen Indonesië en Denemarken zijn de eerste oefenduels richting het EK voetbal volgend jaar zomer bij het Indonesische mannenelftal spelen veel in Nederland geboren spelers ook bij de selectie van de vrouwen wordt daar nu naar gekeken volgens de baas van de Indonesische voetbalbond wordt er binnen afzienbare tijd ook tussen de mannenteams geoefend het weer vanuit het westen komen er nieuwe buien aan bij zon 16 graden tijdens de regen ligt de temperatuur nog wat lager vanavond volgt vanuit het zuidwesten een groot regengebied dan gaat het lange tijd regenen morgen wordt het onstuimig met buien en veel wind dan is het iets warmer bij een graad of 18 dit was het NOS Journaal. That's all for now.",
    
}

def speak(text):
    engine.say(text)
    engine.runAndWait()

def listen():
    with sr.Microphone() as source:
        print("Listening. . .")
        audio = recognizer.listen(source)
        try:
            command = recognizer.recognize_google(audio, language = "nl-NL").lower()
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
    if command in responses:
        speak(responses[command])
    elif command == "exit":
        speak("Goodbye!")
        break

