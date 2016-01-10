from random import randint, shuffle

"""
    Toy task 3: counting
"""

def main():
    existing = set()

    for i in range(100000):
        existing.add(tuple([randint(1, 2) for _ in range(randint(1, 10))]))

    existing = map(list, existing)
    shuffle(existing)

    for X in existing:
        Y = [x for x in X if x == 1]
        print ' '.join(map(str, X)), ';', ' '.join(map(str, Y))


if __name__ == "__main__":
    main()
