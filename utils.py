LEGIBLE = ('legibility', 'legible')
ILLEGIBLE = ('legibility','illegible')

ENGLISH = ('language', 'english')
NOT_ENGLISH = ('language', 'not english')
NA = ('langauge', 'na')

MACHINE_PRINTED = ('class', 'machine printed')
HANDWRITTEN = ('class', 'handwritten')
OTHERS = ('class', 'others')


def inter(list1, list2):
    return list(set(list1) & set(list2))