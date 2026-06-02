import torch
from util import vec_softmax
from transformers import AutoModelForSequenceClassification as amseq, AutoTokenizer as atok

REF = "Jinuuuu/KoELECTRA_fine_tunning_emotion"

class TextModel:
    def __init__(self):
        try:
            self.__model = amseq.from_pretrained(REF)
            self.__model_labels = list(self.__model.config.id2label.values())
            self.__tokenizer = atok.from_pretrained(REF)

        except Exception as e:
            print(f"TextModel Error: {e}")
    
    def predict(self, text: str) -> dict:
        try:
            if not text:
                raise Exception("No text was provided!")
            
            proc_text = self.__tokenizer.encode_plus(text, add_special_tokens = True, return_tensors = 'pt', truncation = True, padding = True)

            with torch.no_grad():
                output = self.__model(**proc_text)
                prob = vec_softmax(output.logits)

            emo = {}

            for i, p in enumerate(prob):
                label = self.__model_labels[i]
                emo[label] = float(p)

            return emo

        except Exception as e:
            print(f"TextModel Error: {e}")

        return dict()
