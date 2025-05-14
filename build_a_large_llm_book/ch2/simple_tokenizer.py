
import urllib.request
import re

class GenerateVocab:
    """ take the sample text and generate a vocab """
    vocab = {}
    
    def __init__(self):
        url = ("https://raw.githubusercontent.com/rasbt/"
               "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
               "the-verdict.txt")
        file_path = "the-verdict.txt"
        urllib.request.urlretrieve(url,file_path)
        
        # preprocess text
        with open("the-verdict.txt", "r", encoding="utf-8") as f:
            raw_text = f.read()
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
         
        #sort and output size
        all_words = sorted(list(set(preprocessed)))
        all_words.extend(["<|endoftext|>", "<|unk|>"])
        
        #generate token list
        self.vocab = {token:integer for integer,token in enumerate(all_words)}
 
    def get_vocab(self):
       return self.vocab
        
class SimpleTokenizer:
    """ Simple Text Tokenizer """

    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}


    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]

        preprocessed = [item if item in self.str_to_int
                        else "<|unk|>" for item in preprocessed]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids


    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?_!"()\'])', r'\1', text)
        return text 

vocab = GenerateVocab()
vocab = vocab.get_vocab()
tokenizer = SimpleTokenizer(vocab)

print("EXAMPLE 1 ENCODED: ")
text = """It's the last thing he painted, you know,"
       Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids) 

print("EXAMPLE 1 DECODED: ")
print(tokenizer.decode(ids))

text2 = "Hello, do you like tea?"
text3 = "In the sunlit terraces of the palace."
text4 = "<|endoftext|> ".join((text2, text3))

print("EXAMPLE 2 ENCODED: ")
ids2 = tokenizer.encode(text4)
print(ids2)

print("EXAMPLE 2 DECODED: ")
print(tokenizer.decode(ids2))
