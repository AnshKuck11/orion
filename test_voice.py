import pyttsx3

engine = pyttsx3.init()

voices = engine.getProperty('voices')

engine.setProperty('rate', 180)
engine.setProperty('volume', 1.0)
engine.setProperty('voice', voices[0].id)

print("About to speak...")
engine.say("Greetings. I am Orion, ready to assist.")
engine.runAndWait()
print("Done speaking!")