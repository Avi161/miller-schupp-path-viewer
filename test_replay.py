"""Test replay logic for presentation 62 (idx 61) - no ac_solver dependency."""
import json

def reduce_word(word):
    result = []
    for c in word:
        if c == 0:
            continue
        if result and (
            (result[-1] == 1 and c == -1) or (result[-1] == -1 and c == 1) or
            (result[-1] == 2 and c == -2) or (result[-1] == -2 and c == 2)
        ):
            result.pop()
            continue
        result.append(c)
    return result

def invert_word(word):
    result = []
    for i in range(len(word) - 1, -1, -1):
        if word[i] == 0:
            continue
        c = word[i]
        result.append(-1 if c == 1 else -2 if c == 2 else 1 if c == -1 else 2)
    return result

def apply_move(r1, r2, action):
    if action == 0:
        return r1, reduce_word(r2 + r1)
    if action == 1:
        return reduce_word(r1 + invert_word(r2)), r2
    if action == 2:
        return r1, reduce_word(r2 + invert_word(r1))
    if action == 3:
        return reduce_word(r1 + r2), r2
    if action == 4:
        return r1, reduce_word([-1] + r2 + [1])
    if action == 5:
        return reduce_word([-2] + r1 + [2]), r2
    if action == 6:
        return r1, reduce_word([-2] + r2 + [2])
    if action == 7:
        return reduce_word([1] + r1 + [-1]), r2
    if action == 8:
        return r1, reduce_word([1] + r2 + [-1])
    if action == 9:
        return reduce_word([2] + r1 + [-2]), r2
    if action == 10:
        return r1, reduce_word([2] + r2 + [-2])
    if action == 11:
        return reduce_word([-1] + r1 + [1]), r2
    return r1, r2

def to_str(w):
    sym = {1: 'x', 2: 'y', -1: 'X', -2: 'Y'}
    return ''.join(sym.get(c, '?') for c in w) if w else 'ε'

def main():
    with open('presentations.json') as f:
        pres = json.load(f)
    with open('solutions.jsonl') as f:
        for line in f:
            row = json.loads(line)
            if row['idx'] == 61:
                path = row['path']
                break
    p = pres[61]
    n = len(p) // 2
    r1 = [x for x in p[:n] if x != 0]
    r2 = [x for x in p[n:] if x != 0]
    print('Initial R1:', to_str(r1))
    print('Initial R2:', to_str(r2))
    print('Path length:', len(path))
    for i, (action, length) in enumerate(path):
        r1, r2 = apply_move(r1, r2, action)
        print(f'After step {i+1} (action {action}): R1={to_str(r1)}, R2={to_str(r2)}, total len={len(r1)+len(r2)}')
    print('Final R1:', to_str(r1))
    print('Final R2:', to_str(r2))
    print('Success:', len(r1) + len(r2) == 2)

if __name__ == '__main__':
    main()
