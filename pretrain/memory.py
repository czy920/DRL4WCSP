import random

class Mem:
    def __init__(self, capacity):
        self.s = [0] * capacity
        self.a = [0] * capacity
        self.r = [0] * capacity
        self.s_prime = [0] * capacity
        self.done = [0] * capacity
        self.capacity = capacity
        self.pointer = 0
        self.size = 0

    def add(self, s, a, r, s_prime, done):
        self.s[self.pointer] = s
        self.a[self.pointer] = a
        self.r[self.pointer] = r
        self.s_prime[self.pointer] = s_prime
        self.done[self.pointer] = done

        self.pointer += 1
        self.size += 1
        self.pointer %= self.capacity
        self.size = min(self.size, self.capacity)

    def sample(self, n):
        n = min(n, self.size)
        idxes = random.sample(range(self.size), n)
        s, a, r, s_prime, done = [], [], [], [], []
        for idx in idxes:
            s.append(self.s[idx])
            a.append(self.a[idx])
            r.append(self.r[idx])
            s_prime.append(self.s_prime[idx])
            done.append(self.done[idx])
        return s, a, r, s_prime, done
