import re

def get_words(tokens):
    return [item[0] for item in tokens]

class Tokenizer:
    def __init__(self):
        self.SPLIT_CHARACTERS = '( |\n|\u21b5|\r|\t|,|!|\\?|;|:|-|–|—|~|_|\\|/|<|>|\^|\(|\)|\[|\]|\\")'
        self.SPLIT_CHARACTERS_INLINE = ' \n\r\t,!?;:-–—~_\\/|<>^()[]\"'
        self.SPLIT_CHARACTERS_DICT = {
            ' ': 1,
            '\n': 1,
            '\r': 1,
            '\t': 1,
            ',': 1,
            '!': 1,
            '?': 1,
            ';': 1,
            ':': 1,
            '-': 1,
            '–': 1,
            '—': 1,
            '~': 1,
            '_': 1,
            '\\': 1,
            '/': 1,
            '|': 1,
            '<': 1,
            '>': 1,
            '^': 1,
            '(': 1,
            ')': 1,
            '[': 1,
            ']': 1,
            '\"': 1
        }
        self.DECIMAL_SPLIT = '.'
        self.DECIMAL_PATTERN = '(\d+\.\d+)'
        self.DEID_SPLIT = '-'
        self.ABBR_PATTERN = 'abbr\|[A-Za-z]+\|'

    # Tokens format: [str, char_offset, word_offset]
    def get_tokens(self, text, offset=0):
        tokens = []
        i = 0
        n = 0

        # tokenize words by space and add non-empty ones
        # text_r = text.encode('ascii', 'ignore').decode('utf-8')
        # print(text_r)
        text_split = re.split('( |\n|\u21b5|\r|\t|,|!|\\?|;|:|–|—|~|_|\\|/|<|>|\^|\(|\)|\[|\]|\\|`")', text)
        text_split_after_abbr = []

        for s in text_split:
            if not s:
                continue

            if '|' in s and not re.match(self.ABBR_PATTERN, s):
                # print(repr(s))
                t = re.split(r'\|', s)
                for ss in t:
                    text_split_after_abbr.append(ss)
            else:
                # print(repr(s))
                text_split_after_abbr.append(s)

        for s in text_split_after_abbr:
            if s not in self.SPLIT_CHARACTERS_DICT and len(s) > 0:
                # if we have a period, check if it is float, otherwise split
                if len(s) > 1 and self.DECIMAL_SPLIT in s and not re.match(self.DECIMAL_PATTERN, s):
                    j = 0
                    tt = re.split(self.DECIMAL_SPLIT, s)
                    for ss in tt:
                        if ss not in self.DECIMAL_SPLIT and len(ss) > 0:
                            new_token = [ss, offset+i+j, n]
                            tokens.append(new_token)
                            n += 1
                        j += len(ss)
                else:
                    new_token = [s, offset+i, n]
                    tokens.append(new_token)
                    n += 1

            i += len(s)
        return tokens

    def _combine_texts(self, txt_list, max_length=80000):
        """
        Combine multiple texts to one text, in order to speed up.

        :param txt_list: List of original text strings
        :param max_length: Maximum length of a combined text
        :return: List of combined texts
        """
        multi_texts = []
        len_count = 0
        texts_combined = []
        for txt in txt_list:
            temp_len_count = len_count + len(txt)

            if temp_len_count >= max_length:
                multi_texts.append(self.combine_splitter.join(texts_combined))
                len_count = 0
                texts_combined = []

            len_count += len(txt)
            texts_combined.append(txt)
        # combine last block of texts
        multi_texts.append(self.combine_splitter.join(texts_combined))
        return multi_texts
