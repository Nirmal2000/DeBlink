MORSE_CODE_DICT = { 'A':'.-', 'B':'-...', 
                    'C':'-.-.', 'D':'-..', 'E':'.', 
                    'F':'..-.', 'G':'--.', 'H':'....', 
                    'I':'..', 'J':'.---', 'K':'-.-', 
                    'L':'.-..', 'M':'--', 'N':'-.', 
                    'O':'---', 'P':'.--.', 'Q':'--.-', 
                    'R':'.-.', 'S':'...', 'T':'-', 
                    'U':'..-', 'V':'...-', 'W':'.--', 
                    'X':'-..-', 'Y':'-.--', 'Z':'--..',}

def decrypt(message): 
  
    # extra space added at the end to access the 
    # last morse code 
    message += ' '
  
    decipher = '' 
    citext = '' 
    try:
        for letter in message: 
    
            # checks for space 
            if (letter != ' '): 
    
                # counter to keep track of space 
                i = 0
    
                # storing morse code of a single character 
                citext += letter 
    
            # in case of space 
            else: 
                # if i = 1 that indicates a new character 
                i += 1
    
                # if i = 2 that indicates a new word 
                if i == 2 :  
                    # adding space to separate words 
                    decipher += ' '
                else: 
    
                    # accessing the keys using their values (reverse of encryption) 
                    decipher += list(MORSE_CODE_DICT.keys())[list(MORSE_CODE_DICT 
                    .values()).index(citext)] 
                    citext = '' 
    except:
        return -1
  
    return decipher