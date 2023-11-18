from torch import nn
from transformers import AutoTokenizer
from transformers import BlipForQuestionAnswering


class PubMedBLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip-vqa-base")
        self.language_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

    def forward(self, image, input_text, labels=None):
        inputs = {
            "pixel_values": image,
            "input_ids": input_text.input_ids,
            "labels": input_text if not labels else labels.input_ids
        }
        loss = self.language_model(**inputs)
        return loss.loss

    def vqa_generate(self, image, questions, min_len=1, max_len=20):
        inputs = {
            "pixel_values": image,
            "input_ids": questions.input_ids,
            "min_length": min_len,
            "max_length": max_len
        }
        return self.language_model.generate(**inputs)

    def vqa_forward(self, image, question, answer):
        return self.forward(image, question, answer)
