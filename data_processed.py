
# CV前半部分  CV_top
import pandas as pd
import re
import json
def fact_extract(text):
    pattern1 = r'审理查明(.*?)(\n</p><p>认定|以上事实|上述犯罪|上述事实|上述的事实|上述刑事部分|当庭举证|本案审理|' \
               r'事实清楚|公诉机关|检察院|本案事实|案件事实|庭审举证|法庭质证|下列证据|证据证实|本院认为|另查明|证明)'
    match = re.search(pattern1, text, re.DOTALL)
    if match:
        fact = match.group(1).strip()
        fact = re.sub(r'\n+', ' ', fact)
        fact = re.sub(r'^[:,：，]', '', fact)
        if len(fact) > 30 and len(fact) < 900:
            return fact
    else:
        pattern2 = r'(检察院指控|机关指控)(.*?)(\n</p><p>认定|以上事实|上述犯罪|上述事实|上述的事实|上述刑事部分|当庭举证|本案审理|' \
                   r'事实清楚|公诉机关|检察院|本案事实|案件事实|庭审举证|法庭质证|下列证据|证据证实|本院认为|另查明|证明)'
        match = re.search(pattern2, text, re.DOTALL)
        if match:
            fact = match.group(1).strip()
            fact = re.sub(r'\n+', ' ', fact)
            fact = re.sub(r'^[:,：，]', '', fact)
            if len(fact) > 30 and len(fact) < 900:
                return fact
        else:
            return None

def count_extract(text):
    match = re.search(r'本院认为(.*?)(?:根据|依照|依据|据此|照|《中华|的解释》|》|第)', text, re.DOTALL)
    if match:
        count_V = match.group(1).strip()
        count_V = re.sub(r'\n+', ' ', count_V)
        count_V = re.sub(r'^[:,：，]', '', count_V)
        if "缓刑" in count_V:
            pattern = r'[^，。！？]*缓刑[^，。！？]*[，。！？]'
            # count_V = re.sub(["适用缓刑","宣告缓刑","判处缓刑","缓刑"], '', count_V)
            count_V = re.sub(pattern, '', count_V)
            # if len(count_V) > 60 and len(count_V) < 350:
            return count_V
        else:
            return count_V


def term_desgen(fac):
    description = '本案'
    for key, value in fac.items():
        if key in ['重伤人数','轻伤人数'] and value != 0:
            description += "{}有{}人，".format(key, value)
        if key in ['重伤一级人数', '重伤二级人数', '轻伤一级人数', '轻伤二级人数', '轻微伤人数'] and value != 0:
            description += "其中{}有{}人，".format(key, value)
        if key == '伤残几级' and value != 0:
            description += "被害人伤残{}级，".format(value)
    return description


import re


def huan_desgen(text, fac):
    description1 = "被告人"
    pattern = re.compile(r"(被告人.*?\n\n)", re.DOTALL)
    match = pattern.search(text)
    if match:
        background = match.group(1).strip()
        description1 += "".join([f"是{key}" for key in ['农民', '务工'] if key in background])
        description1 += "，".join([f"{key}文化。" for key in ['小学', '初中', '高中'] if key in background])

    descriptions1_map = {
        '75+': "年龄超过75岁",
        '相对刑事责任（14-16）': "负相对刑事责任",
        '减轻刑事责任（14-18）': "应减轻刑事责任",
        '又聋又哑，盲人': "是聋哑人",
        '无刑事责任（精神病人）': "是精神病人",
        '（间歇性精神病人）负刑事责任': "是间歇性精神病人"
    }

    description1 += "，".join([desc for key, desc in descriptions1_map.items() if fac.get(key) == 1])
    description1 += "。"
    if description1 == "被告人。":
        description1 = ""

    descriptions2_map = {
        '累犯': "是累犯",
        '初犯、偶犯': "是初犯、偶犯",
        '首要分子': "是首要分子",
        '主犯': "是主犯",
        '从犯': "是从犯",
        '胁从犯': "是胁从犯",
        '教唆犯': "是教唆犯",
        '前科': "有犯罪前科"
    }
    description2 = "被告人"
    description2 += "，".join([desc for key, desc in descriptions2_map.items() if fac.get(key) == 1])
    description2 += "。"
    if description2 == "被告人。":
        description2 = ""

    descriptions3_map = {
        '自首': "自首",
        '准自首': "准自首",
        '坦白': "坦白",
        '揭发同案犯共同犯罪事实': "揭发同案犯共同犯罪事实",
        '认罪认罚': "认罪认罚",
        '当庭自愿认罪': "当庭自愿认罪",
        '抢救被害人': "抢救被害人",
        '立功': "有立功表现",
        '谅解': "取得被害人谅解",
        '和解': "与被害人达成和解",
        '积极赔偿并达成谅解': "极赔偿并与被害人达成和解",
        '积极赔偿（无和解、谅解）': "积极赔偿",

    }
    description3 = "案发后，被告人"
    description3 += "，".join([desc for key, desc in descriptions3_map.items() if fac.get(key) == 1])
    description3 += "。"
    if description3 == "案发后，被告人。":
        description3 = ""

    descriptions4_map = {
        '正当防卫': "正当防卫",
        '防卫过当': "防卫过当",
        '紧急避险': "紧急避险",
        '紧急避险过当': "紧急避险过当",
        '犯罪未遂': "犯罪未遂",
        '犯罪预备': "犯罪预备",
        '犯罪中止': "犯罪中止",
        '预谋、报复他人': "预谋、报复他人",

    }
    description4 = "被告人行为属于"
    description4 += "，".join([desc for key, desc in descriptions4_map.items() if fac.get(key) == 1])
    description4 += "。"
    if description4 == "被告人行为属于。":
        description4 = ""

    return description1+description3+description2+description4
def factor_gen(fac, DES):
    factors = []
    factors_V = []
    keys = ['相对刑事责任（14-16）', '减轻刑事责任（14-18）', '75+', '又聋又哑，盲人', '无刑事责任（精神病人）', '负刑事责任（间歇性精神病人）', '正当防卫', '防卫过当',
           '紧急避险', '紧急避险过当', '结伙、聚众斗殴', '互殴', '持械斗殴', '犯罪未遂', '犯罪预备', '犯罪中止', '预谋，报复他人', '被害人过错', '主犯', '从犯', '首要分子',
           '胁从犯', '教唆犯', '立功', '自首', '准自首', '坦白', '揭发同案犯共同犯罪事实', '认罪认罚', '当庭自愿认罪', '谅解', '和解', '积极赔偿并达成谅解', '积极赔偿并达成和解',
           '积极赔偿（无和解、谅解）', '抢救被害人', '累犯', '前科', '初犯、偶犯']
    factors.extend([key for key in keys if fac.get(key) == 1])
    factors_V.extend([1 if fac.get(key) == 1 else 0 for key in keys])
    factors.extend([key for key in ['农民', '务工'] if key in DES])
    factors.extend([key for key in ['小学', '初中', '高中'] if key in DES])
    factors_V.extend([1 if key in DES else 0 for key in ['农民', '务工', '小学', '初中', '高中']])
    return factors, factors_V

df = pd.read_csv(r'data/data_original.csv')
data_json = {}
with open('data/data_processed_vector.txt', 'w', encoding='utf-8') as f:
    for case_num, case, label_2, term, item in zip(df['案号'], df['text'], df['缓刑哑变量'], df["有期徒刑"], df.iloc[:, 6:61].iterrows()):
        FD = fact_extract(case)
        CV = count_extract(case)
        factor_d = item[1]
        term_DES = term_desgen(factor_d)
        huan_DES = huan_desgen(case, factor_d)
        factors, factor_V = factor_gen(factor_d, huan_DES)
        if term > 36:
            label_1 = 0
        else:
            label_1 = 1
        if FD is not None and CV is not None and case_num not in ['（2013）青刑初字第196号', '（2015）怀刑初字第00394号', '（2018）粤0802刑初7号', '（2019）冀0602刑初133号']:
            data_json = {"case_num": case_num, "case": case, "fact_d": FD, "count_V": CV, "term_DES": term_DES,
                         "huan_DES": huan_DES, "factor": factors, "factor_V": factor_V, "label_1": label_1, "label_2": label_2}
            json_str = json.dumps(data_json, ensure_ascii=False)
            f.write(json_str + "\n")

