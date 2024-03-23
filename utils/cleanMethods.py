import re
def remove_between_at_symbols(input_string):
    re_ = re.sub(r'@@ .*?@@', '', input_string)
    re_ = re.sub(r'//.*?\n', '', re_)
    result = re.sub(r'#.*?\n', '', re_)
    return result

def keep_only_english_and_numbers(text):
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return cleaned_text


def is_valid_document_length(tokens):
    return 3 <= len(tokens) <= 256

def clean_special_tokens(document):
    special_patterns = ['<img', 'https:']
    pattern_regex = '|'.join(re.escape(pattern) for pattern in special_patterns)
    cleaned_document = re.sub(pattern_regex, '', document)
    return cleaned_document

def filter_code_examples(code_examples,model,device):
    filtered_examples = []
    for code in code_examples:
        code = remove_between_at_symbols(code)
        # code = keep_only_english_and_numbers(code)
        code = clean_special_tokens(code)

        tokens_ids = model.tokenize([code], max_length=512, mode="<encoder-only>")
        source_ids = torch.tensor(tokens_ids).to(device)
        tokens_embeddings, func_embedding = model(source_ids)
        filtered_examples.append(tokens_embeddings.tolist())

    return filtered_examples

def filterCodeDiff(codeDiff):
    filtered_examples = []
    for result in codeDiff['OriginalCodeDiff']:
        temp = []
        for re in result:
            code = remove_between_at_symbols(re)
            code = clean_special_tokens(code)
            temp.append(code)
        filtered_examples.append(temp)

    codeDiff['OriginalCodeDiff'] = filtered_examples
    return codeDiff