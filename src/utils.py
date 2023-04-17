
# Deine the help function
def labelencoding(polarity):
    if polarity == 'positive':
        return 0
    elif polarity == 'neutral':
        return 1
    elif polarity == 'negative':
        return 2
    else:
        return polarity

# Split the sentence, use index, divide the sentence into two parts(before the target and after the target)
def rl_split(sentence,index):
    x,y = str(index).split(':')
    left = sentence[:int(x)]
    right = sentence[int(y):]
    return left, right

# Transform the category to easy understand version
def transform_category(category):
    if 'GENERAL' in category or 'MISCELLANEOUS' in category:
        category = category.lower()
        x,y = category.split("#")
        return "What do you think of the " + x + " ?"
    elif 'PRICES' in category:
        category = category.lower()
        x,y = category.split("#")
        return "What do you think of the price of it ?"
    elif category == 'FOOD#QUALITY':
        return "What do you think of the quality of food ?"
    elif category == "FOOD#STYLE_OPTIONS":
        return "What do you think of the food choices ?"
    elif category == 'DRINKS#QUALITY':
        return "What do you think of the drinks ?"
    elif category == 'DRINKS#STYLE_OPTIONS':
        return "What do you think of the drink choices ?"

# Define the function to calculate the max length for tokens automatically
def sequence_length(data, tokenizer):
    token_lengths = []
    for sentence in data['sentence']:
        tokens = tokenizer.encode(sentence, max_length=1000)
        token_lengths.append(len(tokens))
        
    # Add 30 to the max to leave buffer
    return max(token_lengths) + 30

# Tokenize input text
def tokenize_text(text):
    sentence_left = text['sentence_left']
    target = text['target']
    aspect_category = text['aspect_category_processed']
    sentence_right = text['sentence_right']
    input_text = '[CLS] ' + sentence_left + ' [SEP] ' + target + ' [SEP] ' + sentence_right + ' [SEP] ' #+ aspect_category + ' [SEP]'
    return tokenizer.encode_plus(input_text, add_special_tokens=False, max_length=max_length, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt')
