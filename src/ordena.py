"""
PRONTO
Ordena lista em ordem alfabetica humana - http://nedbatchelder.com/blog/200712/human_sorting.html
Fonte: https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
"""

import re


def atoi(text):
    """
    Se for digito retorna em formato integer, caso contrario retorna valor recebido
    """
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    Ordena lista em ordem alfabetica humana
    """
    return [atoi(c) for c in re.split(r'(\d+)', text)]


'''
# Exemplo
a_list=[
    "something1",
    "something12",
    "something17",
    "something2",
    "something25",
    "something29"]

a_list.sort(key=natural_keys)
print(a_list)
'''