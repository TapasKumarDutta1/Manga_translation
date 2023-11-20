import sys
import torch
import numpy as np
from torch.nn import functional as F
import spacy
from utils import put_text_in_box

from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizer
sys.path.append("./Transformer-pytorch/")
from seq2seq.attention_is_all_you_need import Transformer
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, AutoTokenizer
import cv2
import glob
from utils import put_text_in_box
from matplotlib import pyplot as plt
D_MODEL = 768
HEADS = 8
N = 6
device = "cpu"


en_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
jp_tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")

target_pad = en_tokenizer.vocab['[PAD]']
input_pad = jp_tokenizer.vocab['[PAD]']

def translate(model, src, max_len=80, custom_sentence=False):
    
    # Tokenizer
    
    model.eval()
    
    if custom_sentence == True:
        src = jp_tokenizer.batch_encode_plus([src], padding=True, truncation=True, return_tensors='pt')['input_ids'].to(device)
    
    src_mask  = (src != input_pad).unsqueeze(-2)
    e_outputs = model.encode(src, src_mask)

    outputs = torch.zeros(max_len).type_as(src.data)
    outputs[0] = 101
    
    for i in range(1, max_len):    
        trg_mask = np.triu(np.ones((1, i, i)), k=1).astype('uint8')
        trg_mask = torch.autograd.Variable(torch.from_numpy(trg_mask) == 0).to(device)

        out = model.out(
            model.decode(
                outputs[:i].unsqueeze(0),
                e_outputs,
                src_mask,
                trg_mask
            )
        )
        out = F.softmax(out, dim=-1)
        val, ix = out[:, -1].data.topk(1)

        outputs[i] = ix[0][0]
        if ix[0][0] == 102:
            break

    return ' '.join([list(en_tokenizer.vocab)[ix] for ix in outputs[1:i]])
    
def create_models():
    tokenizer = AutoTokenizer.from_pretrained(
        "KoichiYasuoka/bert-base-japanese-char-extended"
    )

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        "google/vit-base-patch16-224-in21k",
        "KoichiYasuoka/bert-base-japanese-char-extended",
    )
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size

    # set beam search parameters
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.max_length = 130
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4
    model = model.from_pretrained("/content/checkpoint")
    device = "cpu"
    translate_model = Transformer(N, HEADS, len(list(jp_tokenizer.vocab)), len(list(en_tokenizer .vocab)), D_MODEL)
    translate_model.load_state_dict(
        torch.load(
            "/content/weights_pretrained.hdf5",
            map_location=torch.device("cpu"),
        )
    )
    return translate_model, model, tokenizer


def detect_and_translate(coords):
    translate_model, model, tokenizer = create_models()
    org = cv2.imread("/content/removed.jpg")
    for path in glob.glob("/content/detected/*"):
        image = cv2.imread(path)
        image = cv2.resize(image, (224, 224)) / 255
        image = torch.tensor(image).permute(2, 0, 1)
        generated_ids = model.generate(torch.unsqueeze(image, 0))
        generated_text = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        text = translate(translate_model, generated_text, custom_sentence=True)
        y1, y2, x1, x2 = coords[path.split("/")[-1].split(".")[0]]
        box_coordinates = (x1, y1, y2 - y1, x2 - x1)
        image_with_text = put_text_in_box(org, text, box_coordinates)

    cv2.imwrite("image_with_text.jpg", image_with_text)
    plt.imshow(image_with_text)
    plt.show()
