from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import soundfile as sf
import torch
import sys

def transcribe(fname, model_name):
    # load model and tokenizer
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    # tokenize
    speech, _ = sf.read(fname)
    input_values = processor([speech,],
                             return_tensors="pt",
                             padding="longest").input_values
    # retrieve logits
    logits = model(input_values).logits
    # take argmax and decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    with open(fname.split('.')[0] + "_transcription.txt", "w") as fd:
        fd.write(transcription.lower())
        
if __name__ == "__main__":
    action = sys.argv[1].lower()
    fname = sys.argv[2]
    if action == "transcribe":
        model_name = "facebook/wav2vec2-base-960h"
        if len(sys.argv) > 3:
            model_name = sys.argv[3]
        transcribe(fname, model_name)