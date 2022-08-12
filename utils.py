import cn2an
import re
import json

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def cn_to_an(string):
    try:
        return str(cn2an.cn2an(string, 'normal'))
    except ValueError:
        return string

def an_to_cn(string):
    try:
        return str(cn2an.an2cn(string))
    except ValueError:
        return string

def str_to_num(string):
    try:
        float_val = float(cn_to_an(string))
        if int(float_val) == float_val:
            return str(int(float_val))
        else:
            return str(float_val)
    except ValueError:
        return None

def str_to_year(string):
    year = string.replace('年', '')
    year = cn_to_an(year)
    if is_float(year) and float(year) < 1900:
        year = int(year) + 2000
        return str(year)
    else:
        return None

def load_json(json_file):
    result = []
    if json_file:
        with open(json_file) as file:
            for line in file:
                result.append(json.loads(line))
    return result

def extract_values_from_text(text):
    values = []
    values += extract_year_from_text(text)
    values += extract_num_from_text(text)
    return list(set(values))

def extract_num_from_text(text):
    CN_NUM = '〇一二三四五六七八九零壹贰叁肆伍陆柒捌玖貮两'
    CN_UNIT = '十拾百佰千仟万萬亿億兆点'
    values = []
    num_values = re.findall(r'[-+]?[0-9]*\.?[0-9]+', text)
    values += num_values

    cn_num_unit = CN_NUM + CN_UNIT
    cn_num_texts = re.findall(
        r'[{}]*\.?[{}]+'.format(cn_num_unit, cn_num_unit), text)
    cn_num_values = [str_to_num(text) for text in cn_num_texts]
    values += [value for value in cn_num_values if value is not None]

    cn_num_mix = re.findall(r'[0-9]*\.?[{}]+'.format(CN_UNIT), text)
    for word in cn_num_mix:
        num = re.findall(r'[-+]?[0-9]*\.?[0-9]+', word)
        for n in num:
            word = word.replace(n, an_to_cn(n))
        str_num = str_to_num(word)
        if str_num is not None:
            values.append(str_num)
    return values

def extract_year_from_text(text):
    values = []
    CN_NUM = '〇一二三四五六七八九零壹贰叁肆伍陆柒捌玖貮两'
    CN_UNIT = '十拾百佰千仟万萬亿億兆点'
    num_year_texts = re.findall(r'[0-9][0-9]年', text)
    values += ['20{}'.format(text[:-1]) for text in num_year_texts]
    cn_year_texts = re.findall(r'[{}][{}]年'.format(CN_NUM, CN_NUM), text)
    cn_year_values = [str_to_year(text) for text in cn_year_texts]
    values += [value for value in cn_year_values if value is not None]
    return values
