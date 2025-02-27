import random

m = random.randint(2, 200)
n = random.randint(3, 400)

list_comprehension = [i % 2 for i in range(random.uniform(random.randint(random.randrange(n, m * 100),random.randrange(m, n * 100)), random.randint(1000, 1200)), 100)]

print(list_comprehension)