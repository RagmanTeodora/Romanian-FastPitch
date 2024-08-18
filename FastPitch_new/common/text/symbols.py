""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. '''
from .cmudict import valid_symbols


# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ['@' + s for s in valid_symbols]


def get_symbols(symbol_set='english_basic'):
    if symbol_set == 'english_basic':
        _pad = '_'
        _punctuation = '!\'(),.:;? '
        _special = '-'
        _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        symbols = list(_pad + _special + _punctuation + _letters) + _arpabet
    elif symbol_set == 'english_basic_lowercase':
        _pad = '_'
        _punctuation = '!\'"(),.:;? '
        _special = '-'
        _letters = 'abcdefghijklmnopqrstuvwxyz'
        symbols = list(_pad + _special + _punctuation + _letters) + _arpabet
    elif symbol_set == 'english_expanded':
        _punctuation = '!\'",.:;? '
        _math = '#%&*+-/[]()'
        _special = '_@©°½—₩€$'
        _accented = 'áçéêëñöøćž'
        _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        symbols = list(_punctuation + _math + _special + _accented + _letters) + _arpabet


    elif symbol_set =='romanian_phoneme':
        _pad = '_'
        _punctuation = '!\'(),:;?-. '
        #A is for padding                                                                                       
        _letters_grapheme = 'A1@CDFGHJKPSXZabdefghijklmnoprstuvwzț'
        symbols = list(_pad)+list(_punctuation) + list(_letters_grapheme)
        print ("LEN SYMBOLS: ", len(symbols))

    elif symbol_set == 'romanian_fullPred':
        _pad = '_'
        _punctuation = '!\'(),:.?- '
        _letters_grapheme = 'ĂÂACDEFGHIJKOPSUXZabdefghijklmnoprstuvwzț1@'
        symbols = list(_pad)+list(_punctuation) + list(_letters_grapheme)
    elif symbol_set == 'romanian_fullPred2':
        _pad = '_'
        _punctuation = '!\'(),.;?- '
        _letters_grapheme = 'ĂÂACDEFGHIJKOPSUXZabdefghijklmnoprstuvwzț1@'
        symbols = list(_pad)+list(_punctuation) + list(_letters_grapheme)

    elif symbol_set == 'nl_graphemes':
        _pad = '_'
        _punctuation = '!\'(),.;?- '
        _letters_grapheme = 'Aabcdefghijklmnoprstuvwxyzéëïü’'
        symbols = list(_pad)+list(_punctuation) + list(_letters_grapheme)
    

    else:
        raise Exception("{} symbol set does not exist".format(symbol_set))

    return symbols


def get_pad_idx(symbol_set='english_basic'):
    if symbol_set in {'english_basic', 'english_basic_lowercase', 'romanian_fullPred2', 'romanian_phoneme', 'nl_graphemes'}:
        return 0
    else:
        raise Exception("{} symbol set not used yet".format(symbol_set))
