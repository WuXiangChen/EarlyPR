def transfor_eval(ls):
    if isinstance(ls,str):
       return eval(ls)
    elif isinstance(ls,list):
        return ls

def iter_to_ls(ls):
    tmp = []
    for l in ls:
        tmp.extend(l)
    return tmp

def count_unique_elements(input_list):
    unique_elements = set(input_list)
    return len(unique_elements)

def filtered_balanced_data(owner_test_Commits):
    owner_test_Commits["data_type_len"] = owner_test_Commits["label"].map(count_unique_elements)
    owner_test_Commits = owner_test_Commits[owner_test_Commits["data_type_len"] == 2]
    owner_test_Commits = owner_test_Commits.drop("data_type_len", axis=1)
    owner_test_Commits.reset_index(drop=True, inplace=True)
    return owner_test_Commits


def compare_hex_strings(item):
    sorted_hex_values = sorted(item)
    return sorted_hex_values


