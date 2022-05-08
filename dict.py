import pandas as pd
import random

import json
import numpy as np
import pandas as pd
from prettytable import PrettyTable
import os
import urllib.request
from os.path import exists

word_top_relation = pd.read_pickle("word_top_relation.pkl")
word_reverse_relation = pd.read_pickle("word_reverse_relation.pkl")
word_similar_relation = pd.read_pickle("word_similar_relation.pkl")
word_antonym = pd.read_pickle("word_antonym.pkl")
word_word_scores = pd.read_pickle("word_word_scores.pkl")
word_defition_complete = pd.read_pickle("word_defition_complete.pkl")
word_min400k_dict = pd.read_pickle("word_min400k_dict.pkl")

cols_dict = pd.read_pickle("cols_dict.pkl")
word_history = []
score = 0


def print_ww(w):
    s = "\n".join([d["posp"] + " " + d["tran"] for d in w])
    print(s)

def ishan(text):
    return all(u'\u4e00' <= char <= u'\u9fff' for char in text)

def find_chinese(chinese):
    result = []
    for word in cols_dict.keys():
        if "collins" in cols_dict[word]:
            flag = False
            for definition in cols_dict[word]["collins"]:
                if chinese in definition["tran"]:
                    result.append(word)
                    flag = True
                    break
                #             if "example" in definition:
                #                 for example in definition["example"]:
                #                     if chinese in example["tran"]:
                #                         result.append(word)
                #                         flag = True
                #                         break
                if flag:
                    break
        # break
    if len(result) == 0:
        for word in word_min400k_dict.keys():
            if word not in result:
                if chinese in word_min400k_dict[word]:
                    result.append(word)
    result = sorted(result, key=lambda s: s.lower())
    print(", ".join(result))

def save_search(word):
    with open("search.txt", "a") as f:
        f.write(word)
        f.write("\n")


def split_line(line):
    line_width = 60
    k = 0
    lines = []
    while k * line_width < len(line):
        lines.append(line[k * line_width:(k + 1) * line_width])
        k += 1
    return "\n".join(lines)


def print_ww_detail(word):
    if word in cols_dict and "collins" in cols_dict[word]:
        w = cols_dict[word]["collins"]
        tab = PrettyTable(["类型", "定义", "例句"])
        for d in w:
            examples = ""
            if "example" in d:
                for ex in d["example"]:
                    examples = examples + split_line(ex["ex"]) + "\n" + ex["tran"] + "\n"
            tab.add_row([d["posp"], split_line(d["def"] + "\n" + d["tran"]), examples])
        print(tab)
        # play voice
        # print(w)
        file = "./cache/" + word + ".mp3"
        if not exists(file) and cols_dict[word]["ph_am_mp3"] != "":
            urllib.request.urlretrieve(cols_dict[word]["ph_am_mp3"], file)
        os.system("afplay " + file)
        try:
            playsound("word.mp3")
        except:
            pass
        # 保存到查询记录
        save_search(word)
        return True
    else:
        if word in word_defition_complete:
            print(word_defition_complete[word])
            # 保存到查询记录
            save_search(word)
            return True
        elif word in word_min400k_dict:
            print(word_min400k_dict[word])
            # 保存到查询记录
            save_search(word)
            return True
    return False


def get_random_word():
    global word_history
    l = len(cols_dict)
    wi = random.randint(0, l)
    w = list(cols_dict.keys())[wi]
    print(w)
    # print_ww(cols_dict[w]["collins"])
    word_history.append(w)
    return w


def get_answers(w):
    # 定义词
    result1 = []
    if w in word_top_relation:
        result1 = [w for w in word_top_relation[w] if w in cols_dict]
        result1 = sorted(result1)
    # 相似词
    result2 = []
    if w in word_similar_relation:
        for ww in word_similar_relation[w]:
            result2.append(ww)
        result2 = sorted(result2)
    # 反向关系词
    result3 = []
    if w in word_reverse_relation:
        # 按照关系强弱排序
        res = sorted(word_reverse_relation[w],
                     key=lambda s: -1 * word_word_scores[s][w] if w in word_word_scores[s] else 0.0)
        for ww in res[:30]:
            if ww in cols_dict:
                result3.append(ww)
    result3 = sorted(result3)
    # 形近词
    result4 = []
    for ww in find_distance_similar(w):
        result4.append(ww)
    result4 = sorted(result4)
    # 反义词
    result5 = []
    if w in word_antonym:
        for ww in word_antonym[w]:
            result5.append(ww)
        result5 = sorted(result5)

    details = {"定义词": result1, "相似词": result2, "反向关系词": result3, "形近词": result4, "反义词": result5}
    total_result = []
    for v in details.values():
        total_result.extend(v)
    total_result = sorted(total_result)
    return total_result, details


def edit_distance(s, t):
    """Edit distance of strings s and t. O(len(s) * len(t)). Prime example of a
    dynamic programming algorithm. To compute, we divide the problem into
    overlapping subproblems of the edit distance between prefixes of s and t,
    and compute a matrix of edit distances based on the cost of insertions,
    deletions, matches and mismatches.
    """
    prefix_matrix = np.zeros((len(s) + 1, len(t) + 1))
    prefix_matrix[:, 0] = list(range(len(s) + 1))
    prefix_matrix[0, :] = list(range(len(t) + 1))
    for i in range(1, len(s) + 1):
        for j in range(1, len(t) + 1):
            insertion = prefix_matrix[i, j - 1] + 1
            deletion = prefix_matrix[i - 1, j] + 1
            match = prefix_matrix[i - 1, j - 1]
            if s[i - 1] != t[j - 1]:
                match += 1  # -- mismatch
            prefix_matrix[i, j] = min(insertion, deletion, match)
    return int(prefix_matrix[i, j])

def LevenshteinDistance(s, t):
    v0 = []
    v1 = []
    n = len(t)
    m = len(s)
    for i in range(0, n+1):
        v0.append(i)
        v1.append(0)
    for i in range(0, m):
        v1[0] = i + 1
        for j in range(0, n):
            deletionCost = v0[j+1] + 1
            insertionCost = v1[j] + 1
            if s[i] == t[j]:
                substitutionCost = v0[j]
            else:
                substitutionCost = v0[j] + 1
            v1[j+1] = min(deletionCost, min(insertionCost, substitutionCost))
        for j in range(0, n + 1):
            temp = v0[j]
            v0[j] = v1[j]
            v1[j] = temp
    return v0[n]

def find_distance_similar(word):
    result = []
    for w in cols_dict.keys():
        if w != word and (
                word.startswith(w) or w.startswith(word) or word.endswith(w) or w.endswith(word) or edit_distance(w,
                                                                                                                  word) <= 2):
            result.append(w)
        if len(result) > 20:
            break
    return result

def search_for_similar(word):
    result = []
    for w in cols_dict.keys():
        if w != word and (w.startswith(word) or LevenshteinDistance(w,word) <= min(2, len(word) / 3)):
            result.append(w)
    return result


def print_detail(details):
    l = 0
    for k in details.keys():
        l = max(l, len(details[k]))
    for i in range(0, l):
        for k in details.keys():
            if len(details[k]) <= i:
                details[k].append("")
    tab = PrettyTable(details.keys())
    for i in range(0, l):
        row = []
        for k in details.keys():
            row.append(details[k][i])
        tab.add_row(row)
    print(tab)


if __name__ == "__main__":
    print("输入单词（英文或者中文）后回车, q 退出, h 显示历史, a 显示相关词, ?单词 查询单词的相近单词")
    current_word = ""  # get_random_word()
    answers, answers_details = [], None  # get_answers(current_word)
    while True:
        iput = input('输入> ')
        if iput == "":
            continue
        if iput == "q":
            exit()
        if iput == "a" and current_word != "":
            answers, answers_details = get_answers(current_word)
            print_detail(answers_details)
            continue
        if iput == 'h':
            print(word_history)
            #print("Total Score " + str(score))
            continue
        if ishan(iput):
            find_chinese(iput)
            continue
        if iput[0] == "?":
            similar_words = search_for_similar(iput[1:])
            print(", ".join(similar_words))
            continue
        if len(iput) != 1:
            # print_ww_detail(cols_dict[iput[1:]]["collins"] )
            if  print_ww_detail(iput):
                current_word = iput
                word_history.append(current_word)
            else:
                similar_words = search_for_similar(iput[1:])
                print(", ".join(similar_words))






