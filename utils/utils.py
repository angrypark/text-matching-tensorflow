import argparse
from bisect import bisect
from queue import Queue
from random import random

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args


class JamoProcessor:
    """JAMOCESS MAKER

    Public methods:
    - word_to_jamo: 단어를 자모로 풀어준다.
    - character_to_jamo: 한 글자를 자모 리스트로 풀어준다.
    """
    def __init__(self):
        self._KOR_BEGIN = 44032
        self._KOR_END = 55203
        self._CHOSUNG_BASE = 588
        self._JUNGSUNG_BASE = 28

        self._CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ',
                              'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        self._JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ',
                               'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
        self._JONGSUNG_LIST = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ',
                               'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ',
                               'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ',
                               'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        self._CHOSUNG_IDX = {v: i for i, v in enumerate(self._CHOSUNG_LIST)}
        self._JUNGSUNG_IDX = {v: i for i, v in enumerate(self._JUNGSUNG_LIST)}
        self._JONGSUNG_IDX = {v: i for i, v in enumerate(self._JONGSUNG_LIST)}

        self._MO_COMBINATION = {'ㅏㅣ': 'ㅐ', 'ㅑㅣ': 'ㅒ', 'ㅓㅣ': 'ㅔ',
                                'ㅕㅣ': 'ㅖ', 'ㅗㅏ': 'ㅘ', 'ㅗㅐ': 'ㅙ',
                                'ㅗㅣ': 'ㅚ', 'ㅜㅓ': 'ㅝ', 'ㅜㅔ': 'ㅞ',
                                'ㅜㅣ': 'ㅟ', 'ㅡㅣ': 'ㅢ', 'ㅣㅓ': 'ㅕ',
                                'ㅣㅛ': 'ㅛ'
                                }

        self._chosung_set = set(self._CHOSUNG_LIST)
        self._jungsung_set = set(self._JUNGSUNG_LIST)
        self._jongsung_set = set(self._JONGSUNG_LIST + ['_'])
        self._valid_jamo_set = self._chosung_set | self._jungsung_set \
                               | set(self._JONGSUNG_LIST[1:])

        self._consonant_assimilation = {
            'ㄱㄴ': 'ㅇㄴ', 'ㄱㄹ': 'ㅇㄴ', 'ㄱㅁ': 'ㅇㅁ', 'ㄱㅇ': '_ㄱ', 'ㄲㄴ': 'ㅇㄴ', 'ㄲㄹ': 'ㅇㄴ', 'ㄲㅁ': 'ㅇㅁ',
            'ㄲㅇ': '_ㄲ', 'ㄳㅇ': 'ㄱㅅ', 'ㄴㄹ': 'ㄹㄹ', 'ㄴㅋ': 'ㅇㅋ', 'ㄵㄱ': 'ㄴㄲ', 'ㄵㄷ': 'ㄴㄸ', 'ㄵㄹ': 'ㄹㄹ',
            'ㄵㅂ': 'ㄴㅃ', 'ㄵㅅ': 'ㄴㅆ', 'ㄵㅇ': 'ㄴㅈ', 'ㄵㅈ': 'ㄴㅉ', 'ㄵㅋ': 'ㅇㅋ', 'ㄵㅎ': 'ㄴㅊ', 'ㄶㄱ': 'ㄴㅋ',
            'ㄶㄷ': 'ㄴㅌ', 'ㄶㄹ': 'ㄹㄹ', 'ㄶㅂ': 'ㄴㅍ', 'ㄶㅈ': 'ㄴㅊ', 'ㄷㄴ': 'ㄴㄴ', 'ㄷㄹ': 'ㄴㄴ', 'ㄷㅁ': 'ㅁㅁ',
            'ㄷㅂ': 'ㅂㅂ', 'ㄷㅇ': '_ㄷ', 'ㄹㄴ': 'ㄹㄹ', 'ㄹㅇ': '_ㄹ', 'ㄺㄴ': 'ㄹㄹ','ㄺㅇ': 'ㄹㄱ', 'ㄻㄴ': 'ㅁㄴ',
            'ㄻㅇ': 'ㄹㅁ', 'ㄼㄴ': 'ㅁㄴ', 'ㄼㅇ': 'ㄹㅂ', 'ㄽㄴ': 'ㄴㄴ', 'ㄽㅇ': 'ㄹㅅ', 'ㄾㄴ': 'ㄷㄴ', 'ㄾㅇ': 'ㄹㅌ',
            'ㄿㄴ': 'ㅁㄴ', 'ㄿㅇ': 'ㄹㅍ', 'ㅀㄴ': 'ㄴㄴ', 'ㅀㅇ': '_ㄹ', 'ㅁㄹ': 'ㅁㄴ', 'ㅂㄴ': 'ㅁㄴ', 'ㅂㄹ': 'ㅁㄴ',
            'ㅂㅁ': 'ㅁㅁ', 'ㅂㅇ': '_ㅂ', 'ㅄㄴ': 'ㅁㄴ', 'ㅄㄹ': 'ㅁㄴ', 'ㅄㅁ': 'ㅁㅁ', 'ㅄㅇ': 'ㅂㅅ', 'ㅅㄴ': 'ㄴㄴ',
            'ㅅㄹ': 'ㄴㄴ', 'ㅅㅁ': 'ㅁㅁ', 'ㅅㅂ': 'ㅂㅂ', 'ㅅㅇ': '_ㅅ', 'ㅆㄴ': 'ㄴㄴ', 'ㅆㄹ': 'ㄴㄴ', 'ㅆㅁ': 'ㅁㅁ',
            'ㅆㅂ': 'ㅂㅂ', 'ㅆㅇ': '_ㅆ', 'ㅇㄹ': 'ㅇㄴ', 'ㅈㅇ': '_ㅈ', 'ㅊㅇ': '_ㅊ', 'ㅋㅇ': '_ㅋ', 'ㅌㅇ': '_ㅌ',
            'ㅍㅇ': '_ㅍ', 'ㄴㄱ': 'ㅇㄱ', 'ㄴㅁ': 'ㅁㅁ', 'ㄴㅂ': 'ㅁㅂ', 'ㄴㅍ': 'ㅁㅍ'
        }

    def _is_kor_char(self, c):
        return self._KOR_BEGIN <= ord(c) <= self._KOR_END

    def is_mo(self, c):
        return c in self._jungsung_set

    def is_jamo(self, c):
        return c in self._valid_jamo_set

    def _is_complete_char(self, jamo_list):
        try:
            cho, jung, jong = jamo_list
        except ValueError:
            return False
        return cho in self._chosung_set \
               and jung in self._jungsung_set \
               and jong in self._jongsung_set

    def word_to_jamo(self, word):
        """단어를 자모로 풀어준다.

        :param word: 단어
        :return: 자모 string
        """
        s = ''
        for char in word:
            jamo = self.character_to_jamo(char)
            if len(jamo) == 3 and jamo[2] == ' ':
                jamo[2] = '_'
            s += ''.join(jamo)
        return s

    # TODO: dependency check. 상관 없으면 종성 ' '을 '_'로 바꿈.
    def character_to_jamo(self, c):
        """한 글자를 자모 리스트로 풀어준다.

        :param c: 한 글자
        :return: [초성, 중성, 종성]
        """

        if not self._is_kor_char(c):
            return [c]

        base = ord(c)
        base -= self._KOR_BEGIN

        cho = base // self._CHOSUNG_BASE
        jung = (base - cho * self._CHOSUNG_BASE) // self._JUNGSUNG_BASE
        jong = (base - cho * self._CHOSUNG_BASE - jung * self._JUNGSUNG_BASE)

        return [self._CHOSUNG_LIST[cho], self._JUNGSUNG_LIST[jung], self._JONGSUNG_LIST[jong]]

    def jamo_to_word(self, jamo_string):
        """자모 리스트를 단어로 묶어준다

        :param jamo_string: 자모 리스트 [초성, 중성, 종성, ..., 초성, 중성, 종성]
        :return: 단어
        """
        string_queue = Queue(maxsize=1000)
        check_queue = list()
        for c in jamo_string:
            string_queue.put(c)

        result = []
        while string_queue.qsize() > 0 or len(check_queue) > 0:
            if self._is_complete_char(check_queue):
                result.append(self.jamo_to_character(check_queue))
                check_queue = list()
            elif len(check_queue) < 3:
                if string_queue.empty():
                    result.append(check_queue[0])
                    del check_queue[0]
                else:
                    check_queue.append(string_queue.get())
            else:
                result.append(check_queue[0])
                del check_queue[0]
        return ''.join(result)

    def jamo_to_character(self, jamo_list):
        """자모 리스트를 한 글자로 묶어준다

        :param jamo_list: 자모 리스트 [초성, 중성, 종성]
        :return: 한 글자
        """
        cho_idx = self._CHOSUNG_IDX[jamo_list[0]]
        jung_idx = self._JUNGSUNG_IDX[jamo_list[1]]
        jong_idx = self._JONGSUNG_IDX.get(jamo_list[2], 0)

        return chr(self._KOR_BEGIN + self._CHOSUNG_BASE*cho_idx + self._JUNGSUNG_BASE*jung_idx + jong_idx)

    def mo_combine(self, mo):
        return self._MO_COMBINATION[mo]

    def consonant_assimilation(self, jamo_string):
        """자음동화를 적용합니다.

        :param jamo_string:
        :return:
        """
        for i in range(int(len(jamo_string) / 3)):
            try:
                _consonant = self._consonant_assimilation[jamo_string[3*i+2:3*i+4]]
            except KeyError:
                _consonant = jamo_string[3*i+2:3*i+4]
            jamo_string = jamo_string[:3*i+2] + _consonant + jamo_string[3*i+4:]
        return jamo_string
