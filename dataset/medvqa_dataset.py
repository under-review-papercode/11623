import os
import re
import json
from PIL import Image
from torch.utils.data import Dataset
from vqaTools.vqaExporter import VqaPreprocessor
from itertools import chain

contractions = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "couldn'tve": "couldn't've", "couldnt've": "couldn't've",
    "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
    "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't",
    "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've", "hes": "he's", "howd": "how'd",
    "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm",
    "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've",
    "itll": "it'll", "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
    "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've",
    "oclock": "o'clock", "oughtnt": "oughtn't", "ow's'at": "'ow's'at", "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've",
    "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll",
    "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've",
    "someone'dve": "someone'd've", "someonell": "someone'll", "someones": "someone's",
    "somethingd": "something'd", "somethingd've": "something'd've", "something'dve": "something'd've",
    "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've",
    "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd",
    "theyd've": "they'd've", "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're",
    "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", "wed've": "we'd've",
    "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll",
    "whatre": "what're", "whats": "what's", "whatve": "what've", "whens": "when's",
    "whered": "where'd", "wheres": "where's", "whereve": "where've", "whod": "who'd",
    "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's",
    "whove": "who've", "whyll": "why'll", "whyre": "why're", "whys": "why's", "wont": "won't",
    "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll",
    "yall'd've": "y'all'd've", "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've",
    "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll": "you'll",
    "youre": "you're", "youve": "you've"
}

manual_map = {'none': '0',
              'zero': '0',
              'one': '1',
              'two': '2',
              'three': '3',
              'four': '4',
              'five': '5',
              'six': '6',
              'seven': '7',
              'eight': '8',
              'nine': '9',
              'ten': '10'}
articles = ['a', 'an', 'the']

period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
comma_strip = re.compile("(\d)(\,)(\d)")
punct = ['.', ';', r"/", '[', ']', '"', '{', '}',
         '(', ')', '=', '+', '\\', '_', '-',
         '>', '<', '@', '`', ',', '?', '!']


def category_patch(sample):

    # SLAKE
    if sample.get("content_type", False):
        sample["category"] = [sample["content_type"].lower()]
        return sample

    # RAD/OVQA
    cat = sample["question_type"].lower()
    res = []
    if any(s in cat for s in ["pres", "prse", "condition"]):
        res.append("presence")
    if "abn" in cat:
        res.append("abnormality")
    if "size" in cat:
        res.append("size")
    if "organ" in cat:
        res.append("organ")
    if "pos" in cat:
        res.append("position")
    if any(s in cat for s in ["attrib", "atrib"]):
        res.append("attribute")
    if "color" in cat:
        res.append("color")
    if "plane" in cat:
        res.append("plane")
    if "modality" in cat:
        res.append("modality")
    if "count" in cat:
        res.append("quantity")
    if "other" in cat and not "attribute" in cat:
        res.append("other")

    sample["category"] = res
    return sample

# Notice that VQA score is the average of 10 choose 9 candidate answers cases
# See http://visualqa.org/evaluation.html
def get_score(occurences):
    if occurences == 0:
        return .0
    elif occurences == 1:
        return .3
    elif occurences == 2:
        return .6
    elif occurences == 3:
        return .9
    else:
        return 1.


def process_punctuation(inText):
    outText = inText
    for p in punct:
        if (p in inText or p + ' ' in inText or ' ' + p in inText) or (re.search(comma_strip, inText) != None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = period_strip.sub("", outText, re.UNICODE)
    return outText


def process_digit_article(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manual_map.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText


def multiple_replace(text, wordDict):
    for key in wordDict:
        text = text.replace(key, wordDict[key])
    return text


def preprocess_answer(answer):
    answer = str(answer)
    answer = process_digit_article(process_punctuation(answer))
    answer = answer.replace(',', '').replace('x ray', 'xray')
    return answer


def pre_question(question: str):

    question = question.lower()

    if "? -yes/no" in question:
        question = question.replace("? -yes/no", "")
    if "? -open" in question:
        question = question.replace("? -open", "")
    if "? - open" in question:
        question = question.replace("? - open", "")

    question = question.replace(',', '').replace('?', '').replace('\'s',
                ' \'s').replace('...', '').replace('x ray', 'x-ray').replace('.', '')
    question = question.rstrip(' ')

    return question


class vqa_dataset(Dataset):
    def __init__(self, ann_file, transform, img_root, eos='[SEP]', split="train", max_ques_words=30):
        self.split = split
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))

        self.transform = transform
        self.img_root = img_root
        self.max_ques_words = max_ques_words
        self.eos = eos
        self.preprocess = VqaPreprocessor()

        self.answer_list = set()
        count = {}
        for ann in self.ann:
            a = self.preprocess.pre_answer(ann["answer"])
            self.answer_list.add(a)
            count[a] = 1 if not count.get(a, False) else count[a] + 1
        self.answer_list = list(self.answer_list)

        # PATCHING CATEGORIES
        for i in range(len(self.ann)):
            self.ann[i] = category_patch(self.ann[i])
        self.category_list = set(chain(*[a["category"] for a in self.ann]))

        self.modality_key = "modality" if "modality" in self.ann[0] else "image_organ"  # slake vs rad
        self.area_list = set([a[self.modality_key] for a in self.ann])

        self.samp_weights = []
        for i in range(len(self.ann)):
            self.samp_weights.append(1 / count[self.preprocess.pre_answer(self.ann[i]["answer"])])

    def __len__(self):
        return len(self.ann)

    def get_samp_weights(self):
        return self.samp_weights

    def get_answerlist(self):
        return self.answer_list

    def get_categories(self):
        return self.category_list

    def get_areas(self):
        return self.area_list

    def __getitem__(self, index):

        ann = self.ann[index]

        filename = 'img_name' if 'img_name' in ann.keys() else "image_name"  # rad vs slake notation
        image_path = os.path.join(self.img_root, ann[filename])

        category = ann["category"]
        area = ann[self.modality_key]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        is_open = ann["answer_type"].lower() == "open"
        question = ann['question']
        question = pre_question(question)
        answer = ann['answer']

        # answer = preprocess_answer(ann['answer'])   # old
        answer = self.preprocess.pre_answer(answer)

        return image, question, answer, is_open, category, area



