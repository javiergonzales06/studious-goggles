import whisper

# Localdeki modeli yükle
model = whisper.load_model("/home/javier/Desktop/HuggingFace/whisper-large-v3")  # veya "./whisper-large-v3" gibi klasör

# Ses dosyasını yükle
audio = whisper.load_audio("audio.mp3")
audio = whisper.pad_or_trim(audio)  # 30 saniyeye kadar kırp/pad

# Mel spectrogram'ı çıkar ve modelin cihazına taşı
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# Dili algıla
_, probs = model.detect_language(mel)
language_code = max(probs, key=probs.get)

print(f"Tespit edilen dil: {language_code}")

def detect_language_from_audio(audio_path, model_path="large-v3"):
    model = whisper.load_model(model_path)
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    return max(probs, key=probs.get)

lang = detect_language_from_audio("audio.mp3", "./whisper-large-v3")
print("Tespit edilen dil:", lang)