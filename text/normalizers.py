import re
from abc import ABC, abstractmethod


class Normalizer(ABC):
    """
    Base class for text normalizer
    """
    def __init__(self):
        pass

    @abstractmethod
    def normalize(self, sentence):
        normalized_text = sentence
        return normalized_text

class DummyNormalizer(Normalizer):
    """
    A dummy normalizer which does nothing.
    """
    def __init__(self):
        pass

    def normalize(self, sentence):
        normalized_sentence = sentence
        return normalized_sentence


class BasicNormalizer(Normalizer):
    def __init__(self):
        pass

    def normalize(self, sentence):
        normalized_sentence = delete_quote(sentence)
        normalized_sentence = sibalizer(normalized_sentence)
        return normalized_sentence


class AdvancedNormalizer(Normalizer):
    def __init(self):
        pass

    def normalize(self, sentence):
        normalized_sentence = sentence.lower()
        normalized_sentence = delete_quote(normalized_sentence)
        normalized_sentence = sibalizer(normalized_sentence)
        normalized_sentence = bad_words_exchanger(normalized_sentence)
        return normalized_sentence


def delete_quote(sentence):
    sentence = sentence.replace("'", '').replace('"', '').strip()
    if sentence.find("10자") > -1:
        sentence = sentence[:sentence.find("10자")]
    return sentence


def sibalizer(sentence):
    r = re.compile('씨[ㄱ-ㅎㅏ-ㅣ0-9]{,3}발|씨[ㄱ-ㅎㅏ-ㅣ0-9]{,3}팔|씨[ㄱ-ㅎㅏ-ㅣ0-9]{,3}빨|\
씨[ㄱ-ㅎㅏ-ㅣ0-9]{,3}벌|씨[ㄱ-ㅎㅏ-ㅣ0-9]{,3}바|씨[ㄱ-ㅎㅏ-ㅣ0-9]{,3}뻘|씨[ㄱ-ㅎㅏ-ㅣ0-9]{,3}펄|\
시[ㄱ-ㅎㅏ-ㅣ0-9]{,3}바|씨[ㄱ-ㅎㅏ-ㅣ0-9]{,3}파|씨[ㄱ-ㅎㅏ-ㅣ0-9]{,3}빠|시[ㄱ-ㅎㅏ-ㅣ0-9]{,3}빠|\
시[ㄱ-ㅎㅏ-ㅣ0-9]{,3}파|시[ㄱ-ㅎㅏ-ㅣ0-9]{,3}발|신[ㄱ-ㅎㅏ-ㅣ0-9]{,3}발|씨[ㄱ-ㅎㅏ-ㅣ0-9]{,3}방|\
ㅅ[ㄱ-ㅎㅏ-ㅣ0-9]{,3}ㅂ|ㅆ[ㄱ-ㅎㅏ-ㅣ0-9]{,3}발|ㅅ[ㄱ-ㅎㅏ-ㅣ0-9]{,3}발|ㅆ[ㄱ-ㅎㅏ-ㅣ0-9]{,3}팔')
    for sibal in r.findall(sentence):
        sentence = sentence.replace(sibal, "시발")
    return sentence


def bad_words_exchanger(sentence):
    bad_words = {'쓰레기': {'쓰레기', '쓰래기', 'ㅆㄹㄱ', '쓰렉', '쓰뤠기', '레기', '쓰렉이'},
                 '병신': {'병신', '븅신', '빙신', 'ㅂㅅ'},
                 '존나': {'존나', '졸라', '조낸', '존내', 'ㅈㄴ', '존니', '좆나', '좆도', '좃도', '좃나'},
                 '뻐큐': {'뻐큐', '뻑큐', '凸'},
                 }
    for bad_word in bad_words:
        r = re.compile('|'.join(bad_words[bad_word]))
        tokens = r.findall(sentence)
        for token in tokens:
            sentence = sentence.replace(token, bad_word)
    return sentence


class MatchingModelNormalizer(Normalizer):
    def __init__(self):
        self.patterns = {"month": re.compile("[0-9]{1,2}월"),
                         "hour": re.compile("[0-9]{1,2}시"),
                         "num": re.compile("\d+"),
                         'url': re.compile('(https?:\/\/|s?ftp:\/\/)?([\da-z\.-_?]+)\.([a-z]{2,6})([\/\w_\.-_?&%]*)*\/?'),
                         'emoticon': re.compile('\(이모티콘\)'),
                         'num': re.compile('\d+'),
                         'repeat': re.compile('(\w)\\1{4,}'),
                         'special': re.compile('([^\w.,^-])\\1{1,}'),
                         'dots': re.compile('([.,])\\1{2,}'),
                         'white': re.compile('\s+'), "english": re.compile(""), 
                         'photo': re.compile("Photo"), 
                         'english': re.compile("[a-zA-Z]+")}

        self.convert_hour = {1: "한시", 2: "두시", 3: "세시", 4: "네시", 5: "다섯시",
                             6: "여섯시", 7: "일곱시", 8: "여덟시", 9: "아홉시", 10: "열시",
                             11: "열한시", 12: "열두시"}

        self.convert_month = {1: "일월", 2: "이월", 3: "삼월", 4: "사월", 5: "오월",
                              6: "육월", 7: "칠월", 8: "팔월", 9: "구월", 10: "십월",
                              11: "십일월", 12: "십이월"}
        
        with open("/home/angrypark/reply_matching_model/data/common_english_words.txt", "r") as f:
            self.common_english_words = [line.strip() for line in f]

    def normalize(self, message, url=True, photo=True, convert_hour=True, convert_month=True, repeat=True, 
                  special=True, dots=True, mask_num=True, emoticon=True, double_space=True, english=True):
        """Normalize Chat Sentences for reply matching model
        :param url: url address를 ' (링크) '로 치환
        :param photo: photo를 ' (사진) '로 치환
        :param convert_hour: 시간의 숫자를 한글로 변환
        :param convert_month: 월의 숫자를 한글로 변환
        :param repeat: 5번 이상 반복되는 글자들을 4개로 변환
        :param special: 3번 이상 반복되는 특수문자를 2개로 변환
        :param dots: 2번 이상 반복되는 . , 을 ...으로 변환
        :param mask_num: 위에서 변환되지 않은 모든 숫자를 N으로 변환
        :param emoticon: '(이모티콘)' 앞 뒤에 한 칸 공백을 넣음
        :param double_space: 빈 칸이 2번 이상 반복될 때 1번으로 변환
        :param english: 자주 사용되는 영어 단어를 제외하고 나머지 단어들은 다 삭제
        """
        result = message

        if message.strip() == 'T_FILE':
            return '(파일)'

        if url:
            result = self.patterns['url'].sub('\s(링크)\s', message.strip())
        
        if photo:
            result = self.patterns['photo'].sub('\s(사진)\s', result)
        
        if convert_hour:
            hour_patterns = self.patterns["hour"].findall(result)
            for hour in hour_patterns:
                num = int(hour[:-1])
                if (num<13) & (num>0):
                    converted_hour = self.convert_hour[num]
                    result = result.replace(hour, converted_hour)
                elif (num<25) & (num>0):
                    converted_hour = self.convert_hour[num-12]
                    result = result.replace(hour, "오후 " + converted_hour)

        if convert_month:
            month_patterns = self.patterns["month"].findall(result)
            for month in month_patterns:
                num = int(month[:-1])
                if (num<13) & (num>0):
                    converted_month = self.convert_month[num]
                    result = result.replace(month, converted_month)

        if repeat:
            result = self.patterns['repeat'].sub('\\1\\1\\1\\1', result)
        
        if special:
            result = self.patterns["special"].sub("\\1", result)
        
        if dots:
            result = self.patterns['dots'].sub('...', result)

        if emoticon:
            result = self.patterns['emoticon'].sub('\s(이모티콘)\s', result)

        if double_space:
            result = self.patterns['white'].sub(' ', result)
            
        if english:
            english_words = self.patterns["english"].findall(result)
            for word in english_words:
                if word.lower() not in self.common_english_words:
                    result = result.replace(word, "")
        
        if mask_num:
            result = self.patterns['num'].sub("N", result)

        return result.strip()