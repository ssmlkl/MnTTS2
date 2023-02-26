""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.
'''
_pad        = 'pad'
_punctuation = ';:,.!?¡¿—…"«»“”&$~()* '
_special = "-_'`"
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'


# Export all symbols:
symbols =  [_pad] + list(_punctuation) + list(_letters)+list(_special)

# Special symbol ids
SPACE_ID = symbols.index(" ")
