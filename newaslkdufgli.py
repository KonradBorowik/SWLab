def check_if_valid_k():
    k = 0

    while not(2 <= int(k) <= 6):
        k = input('Input valid "k"')

    return k


def check_if_valid_n():
    n = 0
    while not (10 <= int(n) <= 49):
        n = input('Input valid "n"')

    return n

k = check_if_valid_k()
n = check_if_valid_n()

print(k)
print(n)
